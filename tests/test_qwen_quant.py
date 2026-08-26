from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shamela_rag.embeddings import qwen as qwen_mod
from shamela_rag.embeddings.qwen import Qwen3EmbeddingProvider, download_qwen_gguf
from shamela_rag.eval.dataset import GoldenExample, GoldenSource
from shamela_rag.eval.qwen_quant import (
    QuantComparisonReport,
    QuantVariantMetrics,
    QuantVariantSpec,
    _cosine,
    _load_chunk_texts,
    _load_eval_chunk_rows,
    _mean_cosine,
    build_recommendation,
    default_variant_specs,
    format_quant_table,
    measure_variant,
    run_quant_retrieval,
    save_eval_chunks_jsonl,
    subsample_eval_chunks,
    write_quant_artifacts,
)


def test_qwen_rejects_invalid_quantization() -> None:
    with pytest.raises(ValueError, match="quantization"):
        Qwen3EmbeddingProvider(quantization="fp8")  # type: ignore[arg-type]


def test_qwen_gguf_requires_path() -> None:
    with pytest.raises(ValueError, match="gguf_path"):
        Qwen3EmbeddingProvider(quantization="gguf")


def test_qwen_gguf_path_only_with_gguf_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gguf_path is only valid"):
        Qwen3EmbeddingProvider(gguf_path=tmp_path / "x.gguf")


def test_qwen_rejects_non_positive_gguf_n_ctx(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gguf_n_ctx"):
        Qwen3EmbeddingProvider(quantization="gguf", gguf_path=tmp_path / "m.gguf", gguf_n_ctx=0)


def test_qwen_int8_load_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _load(
        model_id: str,
        *,
        device: str | None,
        truncate_dim: int | None,
        quantization: str | None = None,
    ) -> object:
        captured["quantization"] = quantization
        captured["device"] = device

        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [1]

        class _Model:
            tokenizer = _Tok()

            def get_sentence_embedding_dimension(self) -> int:
                return 8

            def encode(self, *_a: object, **_k: object) -> list[list[float]]:
                return [[0.1] * 8]

        return _Model()

    monkeypatch.setattr(qwen_mod, "_load_sentence_transformer", _load)
    provider = Qwen3EmbeddingProvider(quantization="int8", dims=8)
    assert provider.quantization == "int8"
    assert provider.dims == 8
    assert provider.query_instruction is not None
    assert provider.tokenizer.count("abc") >= 1
    assert captured["quantization"] == "int8"
    assert captured["device"] is None
    assert provider._model.to() is provider._model
    assert len(provider.embed_documents(["a"])) == 1
    assert len(provider.embed_query("q")) == 8
    provider.close()
    assert provider._model is None


def test_qwen_int4_patches_to_and_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load(*_a: object, **_k: object) -> object:
        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [1]

        class _Model:
            tokenizer = _Tok()

            def get_sentence_embedding_dimension(self) -> int:
                return 4

            def encode(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
                return [[0.5] * 4 for _ in texts]

        return _Model()

    monkeypatch.setattr(qwen_mod, "_load_sentence_transformer", _load)
    provider = Qwen3EmbeddingProvider(quantization="int4", dims=4)
    assert provider.quantization == "int4"
    assert provider.embed_documents([]) == []
    assert provider.embed_documents(["x", "y"]) == [[0.5] * 4, [0.5] * 4]
    provider.close()


def test_bitsandbytes_config_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cfg:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _Torch:
        float16 = "float16"

    import sys
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = _Cfg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "torch", _Torch)  # type: ignore[arg-type]

    int8 = qwen_mod._bitsandbytes_config("int8")
    int4 = qwen_mod._bitsandbytes_config("int4")
    assert int8.kwargs == {"load_in_8bit": True}
    assert int4.kwargs["load_in_4bit"] is True
    assert int4.kwargs["bnb_4bit_quant_type"] == "nf4"


def test_load_sentence_transformer_int8_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _ST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured.update(kwargs)

    import sys
    import types

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _ST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(qwen_mod, "_bitsandbytes_config", lambda mode: {"mode": mode})

    qwen_mod._load_sentence_transformer("m", device="cuda", truncate_dim=8, quantization="int8")
    assert captured["model_id"] == "m"
    assert "device" not in captured
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["trust_remote_code"] is True


def test_gguf_provider_embeds_and_truncates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"x")
    seen: list[str] = []

    class _Llama:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["embedding"] is True
            assert kwargs["n_ctx"] == 32

        def create_embedding(self, text: str) -> dict[str, object]:
            seen.append(text)
            return {"data": [{"embedding": [3.0, 4.0, 0.0, 0.0]}]}

    import sys
    import types

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama  # type: ignore[attr-defined]
    fake.LLAMA_POOLING_TYPE_LAST = 3
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)

    provider = Qwen3EmbeddingProvider(
        quantization="gguf",
        gguf_path=gguf,
        dims=4,
        gguf_n_ctx=32,
        gguf_max_chars=10,
    )
    assert provider.tokenizer.count("abc") >= 1
    vec = provider.embed_documents(["abcdefghijklmnop"])[0]
    assert seen[0] == "abcdefghij"
    assert abs(vec[0] - 0.6) < 1e-6
    assert abs(vec[1] - 0.8) < 1e-6
    q = provider.embed_query("hi")
    assert len(q) == 4
    assert "Instruct:" in seen[-1]
    provider.close()
    assert provider._gguf is None


def test_gguf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        qwen_mod._load_gguf_embedder(tmp_path / "missing.gguf", n_ctx=16)


def test_download_qwen_gguf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "Qwen3-Embedding-8B-Q4_K_M.gguf"
    target.write_bytes(b"gguf")

    def _hf_hub_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == qwen_mod.QWEN3_EMBEDDING_GGUF_REPO_ID
        return str(target)

    import sys
    import types

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert download_qwen_gguf(local_dir=tmp_path) == target


def test_default_variant_specs_include_int8_and_optional_gguf(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    specs = default_variant_specs(
        include_fp16=True,
        include_int4=False,
        gguf_path=gguf,
        device="cuda",
        gguf_n_ctx=256,
    )
    assert [s.name for s in specs] == ["fp16-baseline", "int8", "gguf"]
    assert specs[2].gguf_n_ctx == 256


def test_default_variant_specs_cpu_gguf_only(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    specs = default_variant_specs(
        include_fp16=False,
        include_int8=False,
        include_int4=False,
        gguf_path=gguf,
        gguf_baseline_path=tmp_path / "q8.gguf",
    )
    assert [s.name for s in specs] == ["gguf-q8-baseline", "gguf"]


def test_default_variant_specs_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        default_variant_specs(include_fp16=False, include_int8=False, include_int4=False)


def test_build_recommendation_gguf_without_fp16() -> None:
    text = build_recommendation(
        [
            QuantVariantMetrics(
                name="gguf",
                quantization="gguf",
                load_seconds=3.0,
                peak_rss_mb=5200,
                peak_vram_mb=None,
                mean_embed_ms=40.0,
                mean_cosine_vs_baseline=None,
                embed_count=5,
            )
        ]
    )
    assert "GGUF" in text


def test_build_recommendation_prefers_high_cosine_low_vram() -> None:
    rows = [
        QuantVariantMetrics(
            name="fp16-baseline",
            quantization="none",
            load_seconds=10.0,
            peak_rss_mb=16000,
            peak_vram_mb=15000,
            mean_embed_ms=20.0,
            mean_cosine_vs_baseline=None,
            embed_count=5,
        ),
        QuantVariantMetrics(
            name="int8",
            quantization="int8",
            load_seconds=12.0,
            peak_rss_mb=9000,
            peak_vram_mb=8000,
            mean_embed_ms=25.0,
            mean_cosine_vs_baseline=0.98,
            embed_count=5,
        ),
        QuantVariantMetrics(
            name="int4",
            quantization="int4",
            load_seconds=11.0,
            peak_rss_mb=5000,
            peak_vram_mb=4000,
            mean_embed_ms=30.0,
            mean_cosine_vs_baseline=0.90,
            embed_count=5,
        ),
    ]
    assert build_recommendation(rows).startswith("Recommend int8")


def test_build_recommendation_rejects_low_cosine() -> None:
    text = build_recommendation(
        [
            QuantVariantMetrics(
                name="fp16-baseline",
                quantization="none",
                load_seconds=1.0,
                peak_rss_mb=1.0,
                peak_vram_mb=1.0,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=1,
            ),
            QuantVariantMetrics(
                name="int8",
                quantization="int8",
                load_seconds=1.0,
                peak_rss_mb=1.0,
                peak_vram_mb=1.0,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=0.5,
                embed_count=1,
            ),
        ]
    )
    assert "0.95" in text
    assert "int8" in text


def test_build_recommendation_all_errors() -> None:
    text = build_recommendation(
        [
            QuantVariantMetrics(
                name="int8",
                quantization="int8",
                load_seconds=0.0,
                peak_rss_mb=None,
                peak_vram_mb=None,
                mean_embed_ms=0.0,
                mean_cosine_vs_baseline=None,
                embed_count=0,
                error="ImportError: missing",
            )
        ]
    )
    assert "No quantized variant" in text


def test_format_and_write_quant_artifacts(tmp_path: Path) -> None:
    from shamela_rag.eval.comparison import ComparisonReport
    from shamela_rag.eval.harness import EvalReport
    from shamela_rag.eval.metrics import AggregateScore

    retrieval = ComparisonReport(
        ks=(10,),
        results={
            "gguf": EvalReport(
                per_query=(),
                aggregate=AggregateScore(
                    num_queries=1,
                    num_adversarial=0,
                    recall={10: 0.5},
                    hit={10: 1.0},
                    ndcg={10: 0.3},
                    mrr=0.4,
                    mean_latency_ms=1.0,
                    p95_latency_ms=1.0,
                ),
                ks=(10,),
            )
        },
    )
    report = QuantComparisonReport(
        model_id="Qwen/Qwen3-Embedding-8B",
        variants=[
            QuantVariantMetrics(
                name="fp16-baseline",
                quantization="none",
                load_seconds=1.0,
                peak_rss_mb=100.0,
                peak_vram_mb=None,
                mean_embed_ms=5.0,
                mean_cosine_vs_baseline=None,
                embed_count=2,
            )
        ],
        recommendation="Only fp16-baseline ran.",
        retrieval=retrieval,
    )
    table = format_quant_table(report)
    assert "fp16-baseline" in table
    assert "Dense retrieval" in table
    write_quant_artifacts(tmp_path, report)
    assert (tmp_path / "comparison_table.md").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "retrieval" in metrics


def test_cosine_and_mean_cosine() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="length mismatch"):
        _cosine([1.0], [1.0, 2.0])
    assert _mean_cosine([[1.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(1.0)
    assert _mean_cosine([], []) == 0.0


def test_subsample_eval_chunks_round_robin() -> None:
    rows = [
        {"chunk_id": "10:0", "book_id": 10, "text": "a"},
        {"chunk_id": "10:1", "book_id": 10, "text": "b"},
        {"chunk_id": "20:0", "book_id": 20, "text": "c"},
        {"chunk_id": "20:1", "book_id": 20, "text": "d"},
    ]
    out = subsample_eval_chunks(rows, max_chunks=3)
    assert len(out) == 3
    assert {r["book_id"] for r in out} == {10, 20}
    with pytest.raises(ValueError, match="max_chunks"):
        subsample_eval_chunks(rows, max_chunks=0)


def test_save_and_load_eval_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "1:0", "book_id": 1, "text": "hello"},
        {"chunk_id": "2:0", "book_id": 2, "text": "world"},
    ]
    save_eval_chunks_jsonl(path, rows)
    loaded = _load_eval_chunk_rows(path)
    assert loaded == rows
    texts, books = _load_chunk_texts(path, max_chunks=1)
    assert texts == ["hello"]
    assert books == [1]


def test_load_eval_chunks_rejects_bad_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"chunk_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="book_id/text"):
        _load_eval_chunk_rows(path)


def test_measure_variant_records_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("no weights")

    monkeypatch.setattr(
        "shamela_rag.eval.qwen_quant._build_provider",
        _boom,
    )
    metrics, vectors = measure_variant(
        QuantVariantSpec(name="int8", quantization="int8"),
        ["a"],
    )
    assert vectors is None
    assert metrics.error is not None
    assert "RuntimeError" in metrics.error


def test_measure_variant_success_with_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Prov:
        dims = 2

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "shamela_rag.eval.qwen_quant._build_provider",
        lambda _spec: _Prov(),
    )
    logs: list[str] = []
    metrics, vectors = measure_variant(
        QuantVariantSpec(name="gguf", quantization="gguf"),
        ["a", "b"],
        baseline_vectors=[[1.0, 0.0], [1.0, 0.0]],
        progress=logs.append,
    )
    assert vectors is not None
    assert metrics.mean_cosine_vs_baseline == pytest.approx(1.0)
    assert any("embedded" in line for line in logs)


def test_run_quant_retrieval_ranks_matching_book() -> None:
    dataset = [
        GoldenExample(
            example_id="g1",
            query="q1",
            sources=(
                GoldenSource(
                    book_id=10,
                    shamela_page_id=1,
                    confidence="verified",
                    book_title="t",
                ),
            ),
        )
    ]
    report = run_quant_retrieval(
        dataset,
        ["ignored"],
        [10, 20],
        dense_by_variant={
            "gguf": [[1.0, 0.0], [0.0, 1.0]],
        },
        query_vectors_by_variant={"gguf": {"q1": [1.0, 0.0]}},
        candidate_limit=2,
        ks=(1, 2),
    )
    assert report.results["gguf"].aggregate.recall[1] == pytest.approx(1.0)


def test_compare_qwen_quant_cli_parses_flags(tmp_path: Path) -> None:
    from shamela_rag.cli import build_parser

    args = build_parser().parse_args(
        [
            "compare-qwen-quant",
            "--skip-fp16",
            "--no-int8",
            "--gguf",
            str(tmp_path / "q.gguf"),
            "--gguf-n-ctx",
            "256",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.command == "compare-qwen-quant"
    assert args.skip_fp16 is True
    assert args.no_int8 is True
    assert args.gguf_n_ctx == 256
    assert Path(args.gguf) == tmp_path / "q.gguf"


def test_bitsandbytes_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import = importlib.import_module

    def _boom(name: str, *args: object, **kwargs: object) -> object:
        if name in ("torch", "transformers", "bitsandbytes"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(ImportError, match="bitsandbytes"):
        qwen_mod._bitsandbytes_config("int8")


def test_load_sentence_transformer_rejects_gguf() -> None:
    with pytest.raises(ValueError, match="_load_gguf_embedder"):
        qwen_mod._load_sentence_transformer(
            "m", device=None, truncate_dim=None, quantization="gguf"
        )


def test_load_sentence_transformer_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import = importlib.import_module

    def _boom(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(ImportError, match="shamela-rag\\[qwen\\]"):
        qwen_mod._load_sentence_transformer("m", device="cpu", truncate_dim=None)


def test_load_sentence_transformer_sets_device(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _ST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured.update(kwargs)

    import sys
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _ST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    qwen_mod._load_sentence_transformer("m", device="cpu", truncate_dim=None)
    assert captured["device"] == "cpu"


def test_gguf_import_error_and_n_ctx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    real_import = importlib.import_module
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")

    def _boom(name: str, *args: object, **kwargs: object) -> object:
        if name == "llama_cpp":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        qwen_mod._load_gguf_embedder(gguf, n_ctx=16)

    import sys
    import types

    class _Llama:
        def __init__(self, **kwargs: object) -> None:
            raise TypeError("pooling_type")

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama  # type: ignore[attr-defined]
    fake.LLAMA_POOLING_TYPE_LAST = 3
    monkeypatch.setattr(importlib, "import_module", real_import)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    with pytest.raises(TypeError):
        qwen_mod._load_gguf_embedder(gguf, n_ctx=16)
    with pytest.raises(ValueError, match="n_ctx"):
        qwen_mod._load_gguf_embedder(gguf, n_ctx=0)


def test_gguf_llama_typeerror_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    calls: list[set[str]] = []

    class _Llama:
        def __init__(self, **kwargs: object) -> None:
            calls.append(set(kwargs))
            if "pooling_type" in kwargs:
                raise TypeError("no pooling")
            if "n_batch" in kwargs:
                raise TypeError("no n_batch")

    import sys
    import types

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    assert qwen_mod._load_gguf_embedder(gguf, n_ctx=32, n_threads=2) is not None
    assert len(calls) == 3
    assert "pooling_type" in calls[0]
    assert "pooling_type" not in calls[1]
    assert "n_batch" not in calls[2]


def test_download_gguf_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import = importlib.import_module

    def _boom(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(ImportError, match="huggingface_hub"):
        download_qwen_gguf()


def test_qwen_rejects_bad_batch_and_dims() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Qwen3EmbeddingProvider(batch_size=0)
    with pytest.raises(ValueError, match="dims"):
        Qwen3EmbeddingProvider(dims=0)


def test_gguf_dims_mismatch_and_zero_vector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"x")

    class _Llama:
        def __init__(self, **_k: object) -> None:
            return None

        def create_embedding(self, text: str) -> dict[str, object]:
            if text.startswith("zero"):
                return {"data": [{"embedding": [0.0, 0.0]}]}
            return {"data": [{"embedding": [1.0, 2.0, 3.0]}]}

    import sys
    import types

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)

    provider = Qwen3EmbeddingProvider(quantization="gguf", gguf_path=gguf, dims=2, gguf_n_ctx=16)
    assert provider.embed_documents(["zero"]) == [[0.0, 0.0]]
    with pytest.raises(ValueError, match="dims mismatch"):
        provider.embed_documents(["bad"])
    provider.close()


def test_as_list_vectors_dims_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load(*_a: object, **_k: object) -> object:
        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [1]

        class _Model:
            tokenizer = _Tok()

            def get_sentence_embedding_dimension(self) -> int:
                return 2

            def encode(self, *_a: object, **_k: object) -> list[list[float]]:
                return [[1.0, 2.0, 3.0]]

        return _Model()

    monkeypatch.setattr(qwen_mod, "_load_sentence_transformer", _load)
    provider = Qwen3EmbeddingProvider(dims=2)
    with pytest.raises(ValueError, match="embedding dims mismatch"):
        provider.embed_documents(["x"])


def test_build_recommendation_only_baseline() -> None:
    text = build_recommendation(
        [
            QuantVariantMetrics(
                name="fp16-baseline",
                quantization="none",
                load_seconds=1.0,
                peak_rss_mb=1.0,
                peak_vram_mb=1.0,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=1,
            )
        ]
    )
    assert "Only fp16-baseline ran" in text


def test_build_recommendation_only_int8_no_baseline() -> None:
    text = build_recommendation(
        [
            QuantVariantMetrics(
                name="int8",
                quantization="int8",
                load_seconds=1.0,
                peak_rss_mb=1.0,
                peak_vram_mb=1.0,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=1,
            )
        ]
    )
    assert "Only int8 ran" in text


def test_run_qwen_quant_comparison_probe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag.eval import qwen_quant as qq

    def _measure(
        spec: QuantVariantSpec,
        texts: object,
        *,
        baseline_vectors: object = None,
        progress: object = None,
    ) -> tuple[QuantVariantMetrics, list[list[float]] | None]:
        vecs = [[1.0, 0.0] for _ in range(2)]
        cosine = 1.0 if baseline_vectors is not None else None
        return (
            QuantVariantMetrics(
                name=spec.name,
                quantization=str(spec.quantization or "none"),
                load_seconds=0.1,
                peak_rss_mb=100.0,
                peak_vram_mb=None,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=cosine,
                embed_count=2,
            ),
            vecs,
        )

    monkeypatch.setattr(qq, "measure_variant", _measure)
    report = qq.run_qwen_quant_comparison(
        [
            QuantVariantSpec(name="fp16-baseline", quantization=None),
            QuantVariantSpec(name="int8", quantization="int8"),
        ],
        probe_texts=["a", "b"],
    )
    assert len(report.variants) == 2
    assert report.variants[1].mean_cosine_vs_baseline == pytest.approx(1.0)
    assert "Recommend" in report.recommendation or "int8" in report.recommendation


def test_run_qwen_quant_comparison_chunks_and_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag.eval import qwen_quant as qq

    chunks = tmp_path / "chunks.jsonl"
    save_eval_chunks_jsonl(
        chunks,
        [
            {"chunk_id": "10:0", "book_id": 10, "text": "alpha"},
            {"chunk_id": "20:0", "book_id": 20, "text": "beta"},
        ],
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "g1",
                "query": "q1",
                "expected_sources": [
                    {
                        "internal_book_id": 10,
                        "shamela_page_id": 1,
                        "confidence": "verified",
                        "book_title": "t",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def _measure(
        spec: QuantVariantSpec,
        texts: object,
        *,
        baseline_vectors: object = None,
        progress: object = None,
    ) -> tuple[QuantVariantMetrics, list[list[float]] | None]:
        return (
            QuantVariantMetrics(
                name=spec.name,
                quantization="gguf",
                load_seconds=0.1,
                peak_rss_mb=50.0,
                peak_vram_mb=None,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=2,
            ),
            [[1.0, 0.0], [0.0, 1.0]],
        )

    class _Prov:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def close(self) -> None:
            return None

    monkeypatch.setattr(qq, "measure_variant", _measure)
    monkeypatch.setattr(qq, "_build_provider", lambda _spec: _Prov())
    report = qq.run_qwen_quant_comparison(
        [QuantVariantSpec(name="gguf", quantization="gguf", gguf_path=tmp_path / "x.gguf")],
        chunks_path=chunks,
        golden_path=golden,
        max_chunks=2,
        candidate_limit=2,
    )
    assert report.retrieval is not None
    assert "gguf" in report.retrieval.results


def test_run_qwen_quant_comparison_reuses_chunk_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag.eval import qwen_quant as qq

    cache = tmp_path / "eval_chunks.jsonl"
    save_eval_chunks_jsonl(
        cache,
        [{"chunk_id": "1:0", "book_id": 1, "text": "cached"}],
    )
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"id": "g1", "query": "q", "expected_sources": []}) + "\n",
        encoding="utf-8",
    )
    built: list[str] = []

    def _build(*_a: object, **_k: object) -> list[dict[str, object]]:
        built.append("called")
        return []

    monkeypatch.setattr(qq, "build_golden_eval_chunks", _build)
    monkeypatch.setattr(
        qq,
        "measure_variant",
        lambda *_a, **_k: (
            QuantVariantMetrics(
                name="gguf",
                quantization="gguf",
                load_seconds=0.0,
                peak_rss_mb=1.0,
                peak_vram_mb=None,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=1,
            ),
            [[1.0]],
        ),
    )
    report = qq.run_qwen_quant_comparison(
        [QuantVariantSpec(name="gguf", quantization="gguf")],
        corpus_root=tmp_path,
        golden_path=golden,
        chunk_cache_path=cache,
        max_chunks=1,
    )
    assert built == []
    assert report.variants[0].embed_count == 1


def test_run_qwen_quant_comparison_corpus_requires_golden(tmp_path: Path) -> None:
    from shamela_rag.eval.qwen_quant import run_qwen_quant_comparison

    with pytest.raises(ValueError, match="requires --golden"):
        run_qwen_quant_comparison(
            [QuantVariantSpec(name="int8", quantization="int8")],
            corpus_root=tmp_path,
        )


def test_build_golden_eval_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from shamela_rag.data.discovery import BookLocation
    from shamela_rag.eval import qwen_quant as qq

    book_dir = tmp_path / "1"
    book_dir.mkdir()

    class _Chunk:
        pass

    class _Result:
        chunks = [_Chunk(), _Chunk()]

    monkeypatch.setattr(
        "shamela_rag.data.discovery.iter_valid_books",
        lambda _root: [
            BookLocation(book_dir=book_dir, book_id=1, category_id=1, has_all_files=True)
        ],
    )
    monkeypatch.setattr("shamela_rag.chunking.orchestrator.chunk_book", lambda _d: _Result())
    monkeypatch.setattr("shamela_rag.ingestion.pipeline.dense_input", lambda _c: "chunk-text")

    dataset = [
        GoldenExample(
            example_id="g1",
            query="q",
            sources=(
                GoldenSource(
                    book_id=1,
                    shamela_page_id=1,
                    confidence="verified",
                    book_title="t",
                ),
            ),
        )
    ]
    rows = qq.build_golden_eval_chunks(dataset, tmp_path)
    assert len(rows) == 2
    assert rows[0]["text"] == "chunk-text"
    assert rows[0]["book_id"] == 1


def test_build_golden_eval_chunks_empty_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag.eval import qwen_quant as qq

    monkeypatch.setattr("shamela_rag.data.discovery.iter_valid_books", lambda _root: [])
    with pytest.raises(ValueError, match="no chunks built"):
        qq.build_golden_eval_chunks(
            [
                GoldenExample(
                    example_id="g1",
                    query="q",
                    sources=(
                        GoldenSource(
                            book_id=99,
                            shamela_page_id=None,
                            confidence="verified",
                            book_title="t",
                        ),
                    ),
                )
            ],
            tmp_path,
        )


def test_load_eval_chunks_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no chunks found"):
        _load_eval_chunk_rows(path)


def test_subsample_returns_all_when_under_cap() -> None:
    rows = [{"chunk_id": "1:0", "book_id": 1, "text": "a"}]
    assert subsample_eval_chunks(rows, max_chunks=10) == rows


def test_rss_and_vram_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag.eval import qwen_quant as qq

    class _Mem:
        rss = 2 * 1024 * 1024

    class _Proc:
        def memory_info(self) -> _Mem:
            return _Mem()

    class _Psutil:
        @staticmethod
        def Process() -> _Proc:
            return _Proc()

    import sys
    import types

    monkeypatch.setitem(sys.modules, "resource", types.ModuleType("resource"))
    monkeypatch.setitem(sys.modules, "psutil", _Psutil)  # type: ignore[arg-type]
    assert qq._rss_mb() == pytest.approx(2.0)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def max_memory_allocated() -> int:
            return 3 * 1024 * 1024

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            return None

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _Torch)  # type: ignore[arg-type]
    assert qq._vram_mb() == pytest.approx(3.0)
    qq._reset_cuda_peak()


def test_run_compare_qwen_quant_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from shamela_rag import cli

    out = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [
            "compare-qwen-quant",
            "--skip-fp16",
            "--no-int8",
            "--no-int4",
            "--gguf",
            str(tmp_path / "q.gguf"),
            "--output-dir",
            str(out),
        ]
    )

    fake_report = QuantComparisonReport(
        model_id="Qwen/Qwen3-Embedding-8B",
        variants=[
            QuantVariantMetrics(
                name="gguf",
                quantization="gguf",
                load_seconds=1.0,
                peak_rss_mb=10.0,
                peak_vram_mb=None,
                mean_embed_ms=2.0,
                mean_cosine_vs_baseline=None,
                embed_count=1,
            )
        ],
        recommendation="Use GGUF",
    )

    monkeypatch.setattr(
        "shamela_rag.eval.qwen_quant.run_qwen_quant_comparison",
        lambda *_a, **_k: fake_report,
    )
    assert cli.run_compare_qwen_quant(args) == 0
    assert (out / "metrics.json").is_file()


def test_run_compare_qwen_quant_requires_golden_with_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag import cli

    args = cli.build_parser().parse_args(
        [
            "compare-qwen-quant",
            "--chunks",
            str(tmp_path / "c.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert cli.run_compare_qwen_quant(args) == 1


def test_run_qwen_quant_comparison_force_rechunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag.eval import qwen_quant as qq

    cache = tmp_path / "eval_chunks.jsonl"
    save_eval_chunks_jsonl(
        cache,
        [{"chunk_id": "old:0", "book_id": 1, "text": "stale"}] * 5,
    )
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"id": "g1", "query": "q", "expected_sources": []}) + "\n",
        encoding="utf-8",
    )

    def _build(*_a: object, **_k: object) -> list[dict[str, object]]:
        return [{"chunk_id": f"1:{i}", "book_id": 1, "text": f"new-{i}"} for i in range(4)]

    monkeypatch.setattr(qq, "build_golden_eval_chunks", _build)
    monkeypatch.setattr(
        qq,
        "measure_variant",
        lambda *_a, **_k: (
            QuantVariantMetrics(
                name="gguf",
                quantization="gguf",
                load_seconds=0.0,
                peak_rss_mb=1.0,
                peak_vram_mb=None,
                mean_embed_ms=1.0,
                mean_cosine_vs_baseline=None,
                embed_count=2,
            ),
            [[1.0], [1.0]],
        ),
    )
    report = qq.run_qwen_quant_comparison(
        [QuantVariantSpec(name="gguf", quantization="gguf")],
        corpus_root=tmp_path,
        golden_path=golden,
        chunk_cache_path=cache,
        force_rechunk=True,
        max_chunks=2,
    )
    assert report.variants[0].embed_count == 2
    reloaded = _load_eval_chunk_rows(cache)
    assert reloaded[0]["text"].startswith("new-")


def test_provider_close_clears_cuda_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"x")
    cleared: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            cleared.append("ok")

    class _Torch:
        cuda = _Cuda()

    class _Llama:
        def __init__(self, **_k: object) -> None:
            return None

        def create_embedding(self, text: str) -> dict[str, object]:
            return {"data": [{"embedding": [1.0, 0.0]}]}

    import sys
    import types

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "torch", _Torch)  # type: ignore[arg-type]
    provider = Qwen3EmbeddingProvider(quantization="gguf", gguf_path=gguf, dims=2, gguf_n_ctx=16)
    provider.close()
    assert cleared == ["ok"]


def test_main_dispatches_compare_qwen_quant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag import cli

    monkeypatch.setattr(cli, "run_compare_qwen_quant", lambda _args: 0)
    assert (
        cli.main(
            [
                "compare-qwen-quant",
                "--output-dir",
                str(tmp_path),
                "--skip-fp16",
                "--no-int8",
                "--no-int4",
                "--gguf",
                str(tmp_path / "x.gguf"),
            ]
        )
        == 0
    )
