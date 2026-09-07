"""Tests for experiment pipeline and compilation modules."""

import numpy as np
import pytest

from stp_rag.eval.compilation import compile_results_table, compute_paired_bootstrap
from stp_rag.eval.results_schema import ArmResult, write
from stp_rag.experiments.pipeline import run_ablation_arm


def test_paired_bootstrap_logic():
    # Synthetic case where Arm A is consistently better by ~0.15
    np.random.seed(42)
    arm_a = np.array([0.80, 0.85, 0.82, 0.88, 0.84])
    arm_b = np.array([0.65, 0.70, 0.68, 0.72, 0.69])

    delta, (ci_low, ci_high), sig = compute_paired_bootstrap(arm_a, arm_b)
    assert delta > 0.1
    assert ci_low > 0  # Zero strictly excluded
    assert sig is True


def test_compile_results_table(tmp_path):
    # Write mock ArmResults for two systems
    for seed in [42, 123]:
        r_fixed = ArmResult(
            system="fixed_size",
            seed=seed,
            split="test",
            recall_at_1=0.4,
            recall_at_5=0.6,
            recall_at_10=0.7,
            mrr=0.5,
            em=0.4,
            f1=0.5,
            chunks_per_doc=10.0,
            tokens_per_chunk=200.0,
            build_time_sec=0.5,
            retrieval_latency_ms=10.0,
        )
        r_stp = ArmResult(
            system="full_stp",
            seed=seed,
            split="test",
            recall_at_1=0.6,
            recall_at_5=0.85,
            recall_at_10=0.9,
            mrr=0.75,
            em=0.65,
            f1=0.75,
            chunks_per_doc=8.0,
            tokens_per_chunk=220.0,
            build_time_sec=0.6,
            retrieval_latency_ms=11.0,
        )
        write(r_fixed, results_dir=tmp_path)
        write(r_stp, results_dir=tmp_path)

    summary_df, md_table, latex_table = compile_results_table(results_dir=str(tmp_path), split="test")

    assert len(summary_df) == 2
    assert "Fixed-size" in md_table
    assert "Full STP" in md_table
    assert r"\begin{table*}" in latex_table


def test_run_ablation_arm_mlflow_provenance_and_ragas_sentinel(tmp_path):
    """Verifies that ArmResult captures a valid MLflow run ID and honest null RAGAS sentinels."""
    import json
    # Create mini test split
    mini_doc = [{
        "doc_id": "test_d0",
        "title": "Test Doc",
        "text": "# Section 1\nThis is the first sentence discussing topic kinematics.\nThis is the second sentence.",
        "qa_pairs": [{
            "query_id": "q0",
            "question": "What does the first sentence discuss?",
            "answer": "topic kinematics",
            "gold_sentences": ["This is the first sentence discussing topic kinematics."],
        }]
    }]
    split_path = tmp_path / "mini_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(mini_doc, f)

    res = run_ablation_arm(
        variant="velocity_only",
        seed=42,
        split="test",
        data_path=str(split_path),
        results_dir=str(tmp_path),
        use_in_memory_store=True,
    )

    # Verify real MLflow run ID attached
    assert res.mlflow_run_id != "", "mlflow_run_id must not be empty"
    assert res.ragas_faithfulness is None, "RAGAS should default to explicit None/null sentinel"

    # Verify file on disk has the run ID
    json_path = tmp_path / f"velocity_only__seed42__test.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["mlflow_run_id"] == res.mlflow_run_id
    assert disk_data["ragas_faithfulness"] is None
