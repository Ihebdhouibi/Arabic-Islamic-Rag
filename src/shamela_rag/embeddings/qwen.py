"""Qwen3-Embedding-8B backend for ``EmbeddingProvider``.

Uses sentence-transformers. Queries get Qwen's ``Instruct: …\\nQuery:…`` template;
documents are embedded unchanged. Optional ``[qwen]`` / ``[qwen-quant]`` / ``[llm]``
extras install heavy deps. ``quantization``: None (default), ``int8``/``int4``
(bitsandbytes), or ``gguf`` (llama.cpp).
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

DEFAULT_GGUF_N_CTX = 512
DEFAULT_GGUF_MAX_CHARS = DEFAULT_GGUF_N_CTX * 3

QuantizationMode = Literal["int8", "int4", "gguf"]
SUPPORTED_QUANTIZATIONS: frozenset[str] = frozenset({"int8", "int4", "gguf"})


def format_qwen_query(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def _bitsandbytes_config(mode: Literal["int8", "int4"]) -> Any:
    import importlib

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise ImportError(
            "Qwen int8/int4 load needs bitsandbytes + transformers: "
            'pip install -e ".[qwen,qwen-quant]"'
        ) from exc

    config_cls = transformers.BitsAndBytesConfig
    if mode == "int8":
        return config_cls(load_in_8bit=True)
    return config_cls(
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
    import importlib

    try:
        st = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            'Qwen3EmbeddingProvider requires optional deps: pip install "shamela-rag[qwen]"'
        ) from exc

    kwargs: dict[str, Any] = {}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim

    if quantization in ("int8", "int4"):
        # bitsandbytes needs device_map; do not also pass kwargs["device"].
        kwargs["model_kwargs"] = {
            "quantization_config": _bitsandbytes_config(quantization),
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        kwargs["trust_remote_code"] = True
    elif device is not None:
        kwargs["device"] = device

    return st.SentenceTransformer(model_id, **kwargs)


def _load_gguf_embedder(
    gguf_path: Path,
    *,
    n_ctx: int = DEFAULT_GGUF_N_CTX,
    n_threads: int | None = None,
) -> Any:
    import importlib
    import os

    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise ImportError(
            'Qwen GGUF embedding needs llama-cpp-python: pip install -e ".[llm]"'
        ) from exc
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    if n_ctx <= 0:
        raise ValueError(f"n_ctx must be positive, got {n_ctx}")

    threads = n_threads if n_threads is not None else max(1, (os.cpu_count() or 4) - 1)
    kwargs: dict[str, Any] = {
        "model_path": str(gguf_path),
        "embedding": True,
        "n_ctx": n_ctx,
        "n_threads": threads,
        "n_batch": min(512, n_ctx),
        "verbose": False,
    }
    pooling_last = getattr(llama_cpp, "LLAMA_POOLING_TYPE_LAST", None)
    kwargs["pooling_type"] = 3 if pooling_last is None else pooling_last
    llama_cls = llama_cpp.Llama
    try:
        return llama_cls(**kwargs)
    except TypeError:
        kwargs.pop("pooling_type", None)
        try:
            return llama_cls(**kwargs)
        except TypeError:
            kwargs.pop("n_batch", None)
            return llama_cls(**kwargs)


def download_qwen_gguf(
    *,
    filename: str = QWEN3_EMBEDDING_GGUF_Q4_K_M,
    local_dir: Path | str | None = None,
) -> Path:
    """Download an official Qwen3-Embedding GGUF into ``HF_HOME`` / ``local_dir``."""
    import importlib

    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "GGUF download needs huggingface_hub (installed with the [qwen] extra)"
        ) from exc
    path = hub.hf_hub_download(
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
    def count(self, text: str) -> int:
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
        gguf_n_ctx: int = DEFAULT_GGUF_N_CTX,
        gguf_max_chars: int | None = None,
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
        if gguf_n_ctx <= 0:
            raise ValueError(f"gguf_n_ctx must be positive, got {gguf_n_ctx}")

        self._quantization = quantization
        self._batch_size = batch_size
        self._task_description = task_description
        self._gguf: Any | None = None
        self._model: Any | None = None
        self._gguf_max_chars = (
            gguf_max_chars if gguf_max_chars is not None else max(256, gguf_n_ctx * 3)
        )

        if quantization == "gguf":
            assert gguf_path is not None
            path = gguf_path if isinstance(gguf_path, Path) else Path(gguf_path)
            self._gguf = _load_gguf_embedder(path, n_ctx=gguf_n_ctx)
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
            # encode() may call Module.to(); that breaks device_map / bitsandbytes models.
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
        if self._model is not None:
            del self._model
            self._model = None
        if self._gguf is not None:
            del self._gguf
            self._gguf = None
        gc.collect()
        try:
            import importlib

            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _embed_gguf(self, text: str) -> list[float]:
        assert self._gguf is not None
        clipped = text if len(text) <= self._gguf_max_chars else text[: self._gguf_max_chars]
        result = self._gguf.create_embedding(clipped)
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
