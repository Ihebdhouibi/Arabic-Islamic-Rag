"""Qwen3-Embedding-8B backend for ``EmbeddingProvider``.

Uses sentence-transformers. Query text is formatted with Qwen's official
``Instruct: …\\nQuery:…`` template (documents are embedded unchanged). Optional
``[qwen]`` extra installs the heavy deps; integration tests skip when weights
are absent.

Quantization (issue #135 / M3 quantization):
- ``quantization=None`` — default SentenceTransformer load (fp16/bf16 on GPU).
- ``quantization="int8"`` / ``"int4"`` — bitsandbytes weight load (~2x / ~4x VRAM cut).
- ``quantization="gguf"`` — llama.cpp embedding mode from a local GGUF file (CPU fallback).

Install ``pip install -e ".[qwen,qwen-quant]"`` for int8/int4; GGUF needs the ``[llm]`` extra.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from shamela_rag.chunking.tokens import TokenCounter
from shamela_rag.embeddings.provider import EmbeddingProvider

QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
QWEN3_EMBEDDING_GGUF_REPO_ID = "Qwen/Qwen3-Embedding-8B-GGUF"
QWEN3_EMBEDDING_GGUF_Q4_K_M = "Qwen3-Embedding-8B-Q4_K_M.gguf"
QWEN3_EMBEDDING_GGUF_Q8_0 = "Qwen3-Embedding-8B-Q8_0.gguf"
DEFAULT_EMBEDDING_DIMS = 4096
DEFAULT_TASK_DESCRIPTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

QuantizationMode = Literal["int8", "int4", "gguf"]
SUPPORTED_QUANTIZATIONS: frozenset[str] = frozenset({"int8", "int4", "gguf"})


def format_qwen_query(task_description: str, query: str) -> str:
    """Official Qwen3 embedding query formatting (instruction on queries only)."""
    return f"Instruct: {task_description}\nQuery:{query}"


def _bitsandbytes_config(mode: Literal["int8", "int4"]) -> Any:
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Qwen int8/int4 load needs bitsandbytes + transformers: "
            'pip install -e ".[qwen,qwen-quant]"'
        ) from exc
    try:
        import bitsandbytes  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Qwen int8/int4 load needs bitsandbytes: pip install -e ".[qwen,qwen-quant]"'
        ) from exc

    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_sentence_transformer(
    model_id: str,
    *,
    device: str | None,
    truncate_dim: int | None,
    quantization: QuantizationMode | None = None,
) -> Any:
    if quantization == "gguf":
        raise ValueError("gguf quantization uses _load_gguf_embedder, not SentenceTransformer")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            'Qwen3EmbeddingProvider requires optional deps: pip install "shamela-rag[qwen]"'
        ) from exc

    kwargs: dict[str, Any] = {}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim

    if quantization in ("int8", "int4"):
        # device_map is required for bitsandbytes; do not also set kwargs["device"].
        kwargs["model_kwargs"] = {
            "quantization_config": _bitsandbytes_config(quantization),
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        kwargs["trust_remote_code"] = True
    elif device is not None:
        kwargs["device"] = device

    return SentenceTransformer(model_id, **kwargs)


def _load_gguf_embedder(gguf_path: Path, *, n_ctx: int = 8192) -> Any:
    try:
        from llama_cpp import Llama  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            'Qwen GGUF embedding needs llama-cpp-python: pip install -e ".[llm]"'
        ) from exc
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    # Qwen3-Embedding GGUF docs require last-token pooling for correct vectors.
    kwargs: dict[str, Any] = {
        "model_path": str(gguf_path),
        "embedding": True,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    try:
        import llama_cpp

        pooling_last = getattr(llama_cpp, "LLAMA_POOLING_TYPE_LAST", None)
        if pooling_last is not None:
            kwargs["pooling_type"] = pooling_last
        else:
            kwargs["pooling_type"] = 3  # llama.h: LLAMA_POOLING_TYPE_LAST
    except Exception:  # noqa: BLE001 - older wheels may lack the constant
        kwargs["pooling_type"] = 3
    try:
        return Llama(**kwargs)
    except TypeError:
        kwargs.pop("pooling_type", None)
        return Llama(**kwargs)


def download_qwen_gguf(
    *,
    filename: str = QWEN3_EMBEDDING_GGUF_Q4_K_M,
    local_dir: Path | str | None = None,
) -> Path:
    """Download an official Qwen3-Embedding GGUF into ``HF_HOME`` / ``local_dir``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "GGUF download needs huggingface_hub (installed with the [qwen] extra)"
        ) from exc
    path = hf_hub_download(
        repo_id=QWEN3_EMBEDDING_GGUF_REPO_ID,
        filename=filename,
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(path)


class _HFTokenizerCounter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class _HeuristicCharCounter:
    """Fallback when GGUF has no HF tokenizer exposed."""

    def count(self, text: str) -> int:
        # Rough Arabic-aware heuristic used elsewhere in the project.
        return max(1, (len(text) + 2) // 3)


class Qwen3EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_id: str = QWEN3_EMBEDDING_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        dims: int | None = None,
        quantization: QuantizationMode | None = None,
        gguf_path: Path | str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if dims is not None and dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")
        if quantization is not None and quantization not in SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {sorted(SUPPORTED_QUANTIZATIONS)} or None, "
                f"got {quantization!r}"
            )
        if quantization == "gguf" and gguf_path is None:
            raise ValueError("quantization='gguf' requires gguf_path")
        if quantization != "gguf" and gguf_path is not None:
            raise ValueError("gguf_path is only valid with quantization='gguf'")

        self._quantization = quantization
        self._batch_size = batch_size
        self._task_description = task_description
        self._gguf: Any | None = None
        self._model: Any | None = None

        if quantization == "gguf":
            path = Path(gguf_path) if not isinstance(gguf_path, Path) else gguf_path
            self._gguf = _load_gguf_embedder(path)
            self._tokenizer_counter: TokenCounter = _HeuristicCharCounter()
            self._dims = int(dims) if dims is not None else DEFAULT_EMBEDDING_DIMS
            return

        self._model = _load_sentence_transformer(
            model_id,
            device=device,
            truncate_dim=dims,
            quantization=quantization,
        )
        if quantization in ("int8", "int4"):
            # encode() calls Module.to(device), which breaks device_map / bnb models.
            self._model.to = lambda *args, **kwargs: self._model
        self._tokenizer_counter = _HFTokenizerCounter(self._model.tokenizer)
        reported = self._model.get_sentence_embedding_dimension()
        self._dims = int(reported) if reported is not None else DEFAULT_EMBEDDING_DIMS

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def tokenizer(self) -> TokenCounter:
        return self._tokenizer_counter

    @property
    def query_instruction(self) -> str | None:
        return f"Instruct: {self._task_description}\nQuery:"

    @property
    def quantization(self) -> QuantizationMode | None:
        return self._quantization

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._gguf is not None:
            return [self._embed_gguf(text) for text in texts]
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list_vectors(vectors)

    def embed_query(self, text: str) -> list[float]:
        formatted = format_qwen_query(self._task_description, text)
        if self._gguf is not None:
            return self._embed_gguf(formatted)
        assert self._model is not None
        vectors = self._model.encode(
            [formatted],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list_vectors(vectors)[0]

    def close(self) -> None:
        """Release weights so another variant can load on the same host."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._gguf is not None:
            del self._gguf
            self._gguf = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _embed_gguf(self, text: str) -> list[float]:
        assert self._gguf is not None
        result = self._gguf.create_embedding(text)
        data = result["data"] if isinstance(result, dict) else result.data
        row = data[0]["embedding"] if isinstance(data[0], dict) else data[0].embedding
        vector = [float(x) for x in row]
        if len(vector) != self._dims:
            raise ValueError(
                f"GGUF embedding dims mismatch: got {len(vector)}, expected {self._dims}"
            )
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0.0:
            return [0.0] * self._dims
        return [v / norm for v in vector]

    def _as_list_vectors(self, vectors: Any) -> list[list[float]]:
        rows = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
        out: list[list[float]] = []
        for row in rows:
            vector = [float(x) for x in row]
            if len(vector) != self._dims:
                raise ValueError(
                    f"embedding dims mismatch: got {len(vector)}, expected {self._dims}"
                )
            out.append(vector)
        return out
