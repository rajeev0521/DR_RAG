"""Executes the full 6-arm ablation study across 3 seeds on HotpotQA and QASPER."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from stp_rag.eval.compilation import compile_results_table
from stp_rag.experiments.pipeline import run_ablation_arm
from stp_rag.generation.answer import AnswerGenerator


def preflight_check(model_name: str = "phi3:mini", ollama_url: str = "http://localhost:11434/api/generate") -> None:
    """Pre-flight check that fails loudly if the generator is offline or returning errors."""
    print("[Pre-flight] Verifying LLM Generator health...")
    generator = AnswerGenerator(model_name=model_name, ollama_url=ollama_url)
    if not generator.health_check():
        raise RuntimeError(
            f"Pre-flight generator check failed! Cannot reach Ollama at {ollama_url} "
            f"with model '{model_name}'. Please start Ollama (`ollama serve`) before running experiments."
        )
    sample_ans, _ = generator.generate("What is 1+1?", ["Mathematics: 1+1 equals 2."])
    if sample_ans.startswith("[Ollama Offline:") or sample_ans.startswith("[Ollama Error"):
        raise RuntimeError(f"Pre-flight generator produced error response: {sample_ans}")
    print(f"[Pre-flight] Generator online and responding: '{sample_ans.strip()[:60]}...'")


ARMS = [
    "fixed_size",
    "similarity_threshold",
    "velocity_only",
    "velocity_acceleration",
    "full_stp",
    "full_stp_context",
]
SEEDS = [42, 123, 999]


def run_benchmark_dataset(
    dataset_name: str,
    data_path: str,
    results_dir: str,
    tex_path: str,
    split: str = "test",
) -> None:
    print("\n" + "=" * 75)
    print(f"Executing 6-Arm Ablation Suite on Dataset: [{dataset_name.upper()}] ({data_path})")
    print("=" * 75)
    start_time = time.perf_counter()

    for arm in ARMS:
        print(f"\n---> Running Ablation Arm: [{arm}] on {dataset_name}")
        for s in SEEDS:
            t0 = time.perf_counter()
            res = run_ablation_arm(
                variant=arm,
                seed=s,
                split=split,
                data_path=data_path,
                results_dir=results_dir,
                use_in_memory_store=True,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"     Seed {s:3d} | Recall@5: {res.recall_at_5:.3f} | MRR: {res.mrr:.3f} | "
                f"EM: {res.em:.3f} | F1: {res.f1:.3f} | Chunks: {res.chunks_per_doc:.1f} | "
                f"MLflow ID: {res.mlflow_run_id[:8]} ({elapsed:.1f}s)"
            )

    total_time = time.perf_counter() - start_time
    print("\n" + "-" * 75)
    print(f"[{dataset_name.upper()}] runs completed in {total_time:.1f}s. Compiling results table...")

    summary_df, md_table, latex_table = compile_results_table(results_dir, split=split)
    print("\n" + md_table + "\n")

    p_tex = Path(tex_path)
    p_tex.parent.mkdir(parents=True, exist_ok=True)
    p_tex.write_text(latex_table, encoding="utf-8")
    print(f"Exported LaTeX table to: {p_tex}")


def main():
    parser = argparse.ArgumentParser(description="Run STP-RAG ablation benchmarks.")
    parser.add_argument("--dataset", choices=["all", "hotpotqa", "qasper"], default="all",
                        help="Benchmark dataset to evaluate (default: all).")
    args = parser.parse_args()

    preflight_check()

    if args.dataset in ("all", "hotpotqa"):
        run_benchmark_dataset(
            dataset_name="HotpotQA",
            data_path="data/splits/test.json",
            results_dir="results/",
            tex_path="STP_RAG_Overleaf_Package/table2_generated.tex",
            split="test",
        )

    if args.dataset in ("all", "qasper"):
        run_benchmark_dataset(
            dataset_name="QASPER",
            data_path="data/splits/qasper/test.json",
            results_dir="results/qasper/",
            tex_path="STP_RAG_Overleaf_Package/table2_qasper.tex",
            split="test",
        )


if __name__ == "__main__":
    main()
