"""Master compilation, bootstrap statistical significance, and LaTeX/Markdown formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .results_schema import load_all

SYSTEM_ORDER = [
    "fixed_size",
    "similarity_threshold",
    "velocity_only",
    "velocity_acceleration",
    "full_stp",
    "full_stp_context",
]

SYSTEM_DISPLAY_NAMES = {
    "fixed_size": "Fixed-size",
    "similarity_threshold": "Similarity-threshold",
    "velocity_only": "Velocity-only",
    "velocity_acceleration": "Velocity + acceleration",
    "full_stp": "Full STP",
    "full_stp_context": "Full STP + selective context",
}


def compile_results_table(
    results_dir: str = "results/",
    split: str = "test",
) -> Tuple[pd.DataFrame, str, str]:
    """
    Compiles all JSON results for a split across seeds into Table 2.

    Returns:
        (summary_df, markdown_table, latex_table)
    """
    df = load_all(results_dir)
    if df.empty:
        return pd.DataFrame(), "No results found in results directory.", ""

    # Filter to requested split if multiple exist
    if "split" in df.columns:
        df_split = df[df["split"] == split]
        if not df_split.empty:
            df = df_split

    metrics_to_agg = [
        "recall_at_5",
        "mrr",
        "em",
        "f1",
        "chunks_per_doc",
        "build_time_sec",
        "retrieval_latency_ms",
    ]

    summary_rows = []
    for sys_key in SYSTEM_ORDER:
        sys_data = df[df["system"] == sys_key]
        if sys_data.empty:
            continue

        row_dict = {
            "System": SYSTEM_DISPLAY_NAMES.get(sys_key, sys_key),
            "system_key": sys_key,
            "seeds_count": len(sys_data),
        }

        for m in metrics_to_agg:
            vals = sys_data[m].values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            row_dict[f"{m}_mean"] = mean_val
            row_dict[f"{m}_std"] = std_val
            if len(vals) > 1:
                row_dict[m] = f"{mean_val:.3f} ± {std_val:.3f}"
            else:
                row_dict[m] = f"{mean_val:.3f}"

        summary_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)

    # Build Markdown Table
    md_lines = [
        "| Method | Recall@5 | MRR | EM | F1 | Chunks/doc. | Build time (s) | Retrieval latency (ms) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| {r['System']} | {r.get('recall_at_5', 'TBD')} | {r.get('mrr', 'TBD')} | {r.get('em', 'TBD')} | "
            f"{r.get('f1', 'TBD')} | {r.get('chunks_per_doc', 'TBD')} | {r.get('build_time_sec', 'TBD')} | "
            f"{r.get('retrieval_latency_ms', 'TBD')} |"
        )
    md_table = "\n".join(md_lines)

    # Build LaTeX Table for IEEE Paper
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical evaluation of chunking strategies across 6 ablation arms. Mean $\pm$ standard deviation reported across random seeds.}",
        r"\label{tab:results}",
        r"\begin{tabularx}{\textwidth}{lXXXXXXX}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Recall@5} & \textbf{MRR} & \textbf{EM} & \textbf{F1} & \textbf{Chunks/doc} & \textbf{Build (s)} & \textbf{Latency (ms)} \\",
        r"\midrule",
    ]
    for _, r in summary_df.iterrows():
        latex_lines.append(
            f"{r['System']} & {r.get('recall_at_5', 'TBD')} & {r.get('mrr', 'TBD')} & {r.get('em', 'TBD')} & "
            f"{r.get('f1', 'TBD')} & {r.get('chunks_per_doc', 'TBD')} & {r.get('build_time_sec', 'TBD')} & "
            f"{r.get('retrieval_latency_ms', 'TBD')} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ])
    latex_table = "\n".join(latex_lines)

    return summary_df, md_table, latex_table


def compute_paired_bootstrap(
    arm_a_values: np.ndarray,
    arm_b_values: np.ndarray,
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, Tuple[float, float], bool]:
    """
    Computes paired bootstrap confidence interval:
    Δ = Arm_A - Arm_B
    Returns (mean_delta, (ci_lower, ci_upper), is_statistically_significant)
    """
    np.random.seed(random_seed)
    diffs = arm_a_values - arm_b_values
    n = len(diffs)
    if n == 0:
        return 0.0, (0.0, 0.0), False

    boot_diffs = np.zeros(num_resamples)
    for b in range(num_resamples):
        sample_indices = np.random.randint(0, n, size=n)
        boot_diffs[b] = np.mean(diffs[sample_indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha / 2.0) * 100))
    mean_delta = float(np.mean(diffs))

    # Statistically significant if 0 is strictly outside the CI
    is_significant = bool((ci_lower > 0) or (ci_upper < 0))

    return mean_delta, (ci_lower, ci_upper), is_significant
