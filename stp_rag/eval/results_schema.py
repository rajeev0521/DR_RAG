"""Standard results schema and persistence for all experimental arms."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Union
import pandas as pd


@dataclass
class ArmResult:
    """Canonical results representation matching Table 2 and all tracked metrics."""
    system: str                  # "fixed_size" | "similarity_threshold" | "velocity_only" |
                                 # "velocity_acceleration" | "full_stp" | "full_stp_context"
    seed: int
    split: str                   # "dev" | "test"
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    em: float
    f1: float
    chunks_per_doc: float
    tokens_per_chunk: float
    build_time_sec: float
    retrieval_latency_ms: float
    ragas_faithfulness: float = 0.0
    ragas_answer_relevancy: float = 0.0
    ragas_context_precision: float = 0.0
    ragas_context_recall: float = 0.0
    mlflow_run_id: str = ""
    embedding_model_version: str = "BAAI/bge-m3"
    corpus_version: str = "1.0"


def write(result: ArmResult, results_dir: Union[str, Path] = "results/") -> Path:
    """Writes an ArmResult to a standardized JSON file: <system>__seed<k>__<split>.json."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.system}__seed{result.seed}__{result.split}.json"
    file_path = out_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    return file_path


def load_all(results_dir: Union[str, Path] = "results/") -> pd.DataFrame:
    """Loads all arm result JSON files from the results directory into a pandas DataFrame."""
    p_dir = Path(results_dir)
    if not p_dir.exists():
        return pd.DataFrame()

    records: List[dict] = []
    for f in p_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                records.append(data)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
