from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shamela_rag.embeddings import qwen as qwen_mod
from shamela_rag.embeddings.qwen import Qwen3EmbeddingProvider
from shamela_rag.eval.qwen_quant import (
    QuantComparisonReport,
    QuantVariantMetrics,
    QuantVariantSpec,
    build_recommendation,
    default_variant_specs,
    format_quant_table,
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


def test_qwen_int8_load_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _load(
        model_id: str,
        *,
        device: str | None,
        truncate_dim: int | None,
        quantization: str | None = None,
    ) -> object:
        captured["model_id"] = model_id
        captured["device"] = device
        captured["truncate_dim"] = truncate_dim
        captured["quantization"] = quantization

        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [1]

        class _Model:
            tokenizer = _Tok()

            def get_sentence_embedding_dimension(self) -> int:
                return 8

        return _Model()

    monkeypatch.setattr(qwen_mod, "_load_sentence_transformer", _load)
    provider = Qwen3EmbeddingProvider(quantization="int8", dims=8)
    assert provider.quantization == "int8"
    assert captured["quantization"] == "int8"
    assert captured["device"] is None
    assert provider._model.to() is provider._model
    provider.close()
    assert provider._model is None


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


def test_default_variant_specs_include_int8_and_optional_gguf(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    specs = default_variant_specs(
        include_fp16=True,
        include_int4=False,
        gguf_path=gguf,
        device="cuda",
    )
    assert [s.name for s in specs] == ["fp16-baseline", "int8", "gguf"]
    assert specs[0].quantization is None
    assert specs[1].quantization == "int8"
    assert specs[2].gguf_path == gguf


def test_default_variant_specs_cpu_gguf_only(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    specs = default_variant_specs(
        include_fp16=False,
        include_int8=False,
        include_int4=False,
        gguf_path=gguf,
    )
    assert [s.name for s in specs] == ["gguf"]


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
    assert "16GB" in text or "CPU" in text


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
    text = build_recommendation(rows)
    assert text.startswith("Recommend int8")


def test_format_and_write_quant_artifacts(tmp_path: Path) -> None:
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
    )
    table = format_quant_table(report)
    assert "fp16-baseline" in table
    assert "Recommendation:" in table
    write_quant_artifacts(tmp_path, report)
    assert (tmp_path / "comparison_table.md").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_quant_variant_spec_fields() -> None:
    spec = QuantVariantSpec(name="int8", quantization="int8")
    assert spec.name == "int8"
    assert spec.gguf_path is None
