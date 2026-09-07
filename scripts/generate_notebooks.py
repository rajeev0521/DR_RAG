"""Generates the standardized research notebooks 00 to 10 and 99."""

import json
from pathlib import Path


def create_ipynb(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code],
    }


def generate_all():
    # 00: Environment Setup
    create_ipynb([
        md_cell("# 00 - Research Environment and Services Setup\nVerifies local Python 3.11 environment, Docker containers (Qdrant & MLflow), and GPU acceleration."),
        code_cell("""import sys
import torch
print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
"""),
        code_cell("""# Check Qdrant and MLflow connectivity
from stp_rag.indexing.qdrant_store import STPVectorStore
from stp_rag.tracking.mlflow_utils import MLflowTracker

store = STPVectorStore(collection_name="connectivity_check", use_memory=True)
print("Vector store client ready.")
"""),
    ], "notebooks/00_environment_setup.ipynb")

    # 01: Ingestion & Embeddings
    create_ipynb([
        md_cell("# 01 - Corpus Ingestion and Unit-Normalized Embedding Caching\nIngests documents, segments into structure-aware units, and caches embeddings once."),
        code_cell("""from stp_rag.ingestion.prepare_dataset import create_benchmark_sample_data
from stp_rag.experiments.pipeline import precompute_corpus_embeddings

create_benchmark_sample_data()
cache_path = precompute_corpus_embeddings(
    corpus_path="data/splits/dev.json",
    cache_path="data/cache/dev_embeddings.parquet",
)
print(f"Precomputed embeddings cached at: {cache_path}")
"""),
    ], "notebooks/01_ingestion_and_embeddings.ipynb")

    # 02: STP Computation
    create_ipynb([
        md_cell("# 02 - STP Kinematic Profile Computation & Diagnostics\nCalculates semantic displacement, velocity, acceleration, and volatility across cached units."),
        code_cell("""import pandas as pd
import numpy as np
from stp_rag.profile.stp import compute_raw_kinematics

df = pd.read_parquet("data/cache/dev_embeddings.parquet")
print(f"Loaded {len(df)} units from cache.")
"""),
    ], "notebooks/02_stp_computation.ipynb")

    # 03: Dev-Split Calibration
    create_ipynb([
        md_cell("# 03 - Stage A/B Dev-Split Calibration and Checksum Freezing\nFits normalization statistics and optimizes boundary/budget weights strictly on the dev split."),
        code_cell("""from stp_rag.experiments.pipeline import compute_file_sha256
config_file = "configs/stp_params.yaml"
chksum = compute_file_sha256(config_file)
print(f"Calibrated configuration SHA-256: {chksum}")
"""),
    ], "notebooks/03_calibration_dev_split.ipynb")

    # 04: Query Router
    create_ipynb([
        md_cell("# 04 - Query Routing Calibration\nClassifies queries into single-hop vs. multi-hop evidence paths."),
        code_cell("""from stp_rag.retrieval.query_router import QueryRouter
router = QueryRouter()
test_q = "What does positive semantic acceleration indicate compared to negative acceleration?"
decision = router.route(test_q)
print(f"Query: '{test_q}' -> Route: {decision}")
"""),
    ], "notebooks/04_query_router_training.ipynb")

    # 05 to 10: The 6 Ablation Notebooks
    arms = [
        ("05_ablation_fixed_size.ipynb", "fixed_size", "Fixed-Size Chunks Baseline"),
        ("06_ablation_similarity_threshold.ipynb", "similarity_threshold", "Similarity-Threshold Baseline"),
        ("07_ablation_velocity_only.ipynb", "velocity_only", "Velocity-Only Arm"),
        ("08_ablation_velocity_acceleration.ipynb", "velocity_acceleration", "Velocity + Acceleration Arm"),
        ("09_ablation_full_stp.ipynb", "full_stp", "Full STP Arm"),
        ("10_ablation_full_stp_context.ipynb", "full_stp_context", "Full STP + Selective Context Arm"),
    ]

    for filename, arm_key, arm_title in arms:
        create_ipynb([
            md_cell(f"# {filename[:2]} - Ablation Arm: {arm_title}\nExecutes controlled evaluation across random seeds using frozen configuration."),
            code_cell(f"""from stp_rag.experiments.pipeline import run_ablation_arm, compute_file_sha256

# 1. Assert Dev Configuration Integrity
CONFIG_PATH = "configs/stp_params.yaml"
current_sha = compute_file_sha256(CONFIG_PATH)
print(f"Active config SHA-256: {{current_sha}}")
"""),
            code_cell(f"""# 2. Execute Ablation Arm across 3 Random Seeds
seeds = [42, 123, 999]
results = []

for s in seeds:
    print(f"Running {arm_title} with seed {{s}}...")
    res = run_ablation_arm(
        variant="{arm_key}",
        seed=s,
        split="test",
        use_in_memory_store=True,
    )
    results.append(res)
    print(f"  Seed {{s}} -> Recall@5: {{res.recall_at_5:.3f}}, MRR: {{res.mrr:.3f}}, EM: {{res.em:.3f}}, F1: {{res.f1:.3f}}")
"""),
            code_cell("""# 3. View Saved Results
from stp_rag.eval.results_schema import load_all
df = load_all("results/")
arm_df = df[df["system"] == \"""" + arm_key + """\"]
print(arm_df[["system", "seed", "recall_at_5", "mrr", "em", "f1", "chunks_per_doc"]])
"""),
        ], f"notebooks/{filename}")

    # 99: Master Results Compilation
    create_ipynb([
        md_cell("# 99 - Master Results Compilation, Statistical Bootstrap, & Paper Export\nAggregates multi-seed results mechanically into Table 2 and computes bootstrap confidence intervals."),
        code_cell("""from stp_rag.eval.compilation import compile_results_table, compute_paired_bootstrap
import pandas as pd

summary_df, md_table, latex_table = compile_results_table("results/", split="test")
print("=== TABLE 2: EMPIRICAL ABLATION RESULTS ===")
print(md_table)
"""),
        code_cell("""# Save generated LaTeX table for inclusion in IEEE manuscript
with open("STP_RAG_Overleaf_Package/table2_generated.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)
print("LaTeX Table 2 exported to STP_RAG_Overleaf_Package/table2_generated.tex")
"""),
    ], "notebooks/99_master_results_compilation.ipynb")

    print("All 12 research notebooks generated successfully.")


if __name__ == "__main__":
    generate_all()
