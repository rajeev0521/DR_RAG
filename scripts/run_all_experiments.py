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


def main():
    print("=" * 70)
    print("Starting STP-RAG 6-Arm Empirical Ablation Execution")
    print("=" * 70)
    preflight_check()
    start_total = time.perf_counter()

    for arm in ARMS:
        print(f"\n---> Running Ablation Arm: [{arm}]")
        for s in SEEDS:
            t0 = time.perf_counter()
            res = run_ablation_arm(
                variant=arm,
                seed=s,
                split="test",
                use_in_memory_store=True,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"     Seed {s:3d} | Recall@5: {res.recall_at_5:.3f} | MRR: {res.mrr:.3f} | "
                f"Chunks: {res.chunks_per_doc:.1f} | Latency: {res.retrieval_latency_ms:.1f}ms ({elapsed:.2f}s)"
            )

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 70)
    print(f"All runs completed in {total_time:.2f} seconds.")
    print("Compiling Results into Table 2...")
    print("=" * 70)

    summary_df, md_table, latex_table = compile_results_table("results/", split="test")
    print("\n" + md_table + "\n")

    # Export to Overleaf folder
    tex_path = "STP_RAG_Overleaf_Package/table2_generated.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"Exported LaTeX Table 2 to: {tex_path}")


if __name__ == "__main__":
    main()
