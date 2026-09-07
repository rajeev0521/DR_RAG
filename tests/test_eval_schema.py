"""Unit tests for ArmResult schema and disk persistence."""

import tempfile
from pathlib import Path
import pytest

from stp_rag.eval.results_schema import ArmResult, write, load_all


def test_results_schema_write_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = ArmResult(
            system="full_stp",
            seed=42,
            split="test",
            recall_at_1=0.45,
            recall_at_5=0.82,
            recall_at_10=0.91,
            mrr=0.61,
            em=0.55,
            f1=0.74,
            chunks_per_doc=14.2,
            tokens_per_chunk=185.0,
            build_time_sec=1.45,
            retrieval_latency_ms=12.5,
        )

        out_path = write(res, results_dir=tmpdir)
        assert Path(out_path).exists()

        import json
        with open(out_path, "r", encoding="utf-8") as fp:
            saved_json = json.load(fp)
        assert saved_json["ragas_faithfulness"] is None

        df = load_all(tmpdir)
        assert len(df) == 1
        assert df.iloc[0]["system"] == "full_stp"
        assert df.iloc[0]["recall_at_5"] == 0.82
        assert df.iloc[0]["seed"] == 42
