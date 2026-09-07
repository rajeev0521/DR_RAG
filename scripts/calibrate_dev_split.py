"""Two-Stage Calibration Procedure (§6.2 of Methodology Plan).

Stage A: Closed-form median/IQR normalization fit strictly over dev split.
Stage B: Optuna Bayesian refinement of boundary, budget, horizon, and ranking weights
         optimizing dev-split Recall@5.
Stage C: Freeze winning Theta to configs/stp_params.yaml and write configs/stp_params.sha256.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import optuna
import yaml

from stp_rag.chunking.variants import build_chunks_for_variant
from stp_rag.eval.retrieval_metrics import compute_recall_at_k
from stp_rag.indexing.qdrant_store import STPVectorStore
from stp_rag.ingestion.structure_units import extract_structure_aware_units
from stp_rag.profile.embed import EmbeddingModelWrapper
from stp_rag.profile.stp import compute_raw_kinematics, compute_stp, fit_normalization_stats
from stp_rag.profile.structural_features import compute_structural_score
from stp_rag.retrieval.stage1_dense import DenseRetriever

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_calibration(
    dev_path: str = "data/splits/dev.json",
    config_out: str = "configs/stp_params.yaml",
    n_trials: int = 40,
    seed: int = 42,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
) -> Tuple[Dict[str, Any], str]:
    print("=" * 70)
    print("STP-RAG Calibration (§6.2): Fitting Theta on Dev Split")
    print("=" * 70)

    # 1. Load Dev Data
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_docs = json.load(f)

    # Sub-sample for calibration if large corpus
    sample_docs = dev_docs[:50]
    print(f"Ingesting and embedding {len(sample_docs)} dev documents for calibration...")

    embedder = EmbeddingModelWrapper(model_name=embedding_model_name)

    doc_data = []
    v_all_list = []
    a_all_list = []
    sigma_all_list = []
    s_min_vals = []
    s_max_vals = []

    for doc in sample_docs:
        units = extract_structure_aware_units(doc["text"])
        unit_texts = [u.text for u in units]
        embs = embedder.embed_texts(unit_texts)
        cum_toks = np.array([u.cumulative_tokens for u in units], dtype=np.int32)
        s_score, s_min, s_max = compute_structural_score(units)
        s_min_vals.append(s_min)
        s_max_vals.append(s_max)

        d, delta_p, v, a, sigma = compute_raw_kinematics(embs, cum_toks, window_w=5)
        v_all_list.extend(v.tolist())
        a_all_list.extend(a.tolist())
        sigma_all_list.extend(sigma.tolist())

        doc_data.append({
            "doc_id": doc["doc_id"],
            "units": units,
            "embs": embs,
            "cum_toks": cum_toks,
            "s_score": s_score,
            "qa_pairs": doc.get("qa_pairs", []),
        })

    # --- Stage A: Normalization Fit (Closed-Form, Dev Only) ---
    print("\n[Stage A] Fitting closed-form robust normalization statistics (median, IQR)...")
    norm_stats = fit_normalization_stats(
        np.array(v_all_list, dtype=np.float32),
        np.array(a_all_list, dtype=np.float32),
        np.array(sigma_all_list, dtype=np.float32),
    )
    s_min_global = float(np.min(s_min_vals)) if s_min_vals else -0.60
    s_max_global = float(np.max(s_max_vals)) if s_max_vals else 1.80
    structural_scaling = {"min_val": s_min_global, "max_val": s_max_global}

    print("Stage A Normalization Statistics:")
    for k, val in norm_stats.items():
        print(f"  {k}: {val:.4f}")

    # Compute dev STPs using frozen Stage A stats
    for item in doc_data:
        item["stp"] = compute_stp(
            embeddings=item["embs"],
            cumulative_tokens=item["cum_toks"],
            structural_score=item["s_score"],
            norm_stats=norm_stats,
            window_w=5,
        )

    # --- Stage B: Optuna Bayesian Search for Continuous Weights ---
    print(f"\n[Stage B] Running Optuna Bayesian search ({n_trials} trials) optimizing dev Recall@5...")

    def objective(trial: optuna.Trial) -> float:
        w_v = trial.suggest_float("w_v", 0.6, 1.8)
        w_a = trial.suggest_float("w_a", 0.2, 1.2)
        w_sigma = trial.suggest_float("w_sigma", 0.2, 1.2)
        w_s = trial.suggest_float("w_s", 0.3, 1.5)
        b = trial.suggest_float("b", -1.8, 0.2)
        tau_B = trial.suggest_float("tau_B", 0.35, 0.65, step=0.05)

        alpha = trial.suggest_float("alpha", 0.2, 0.8)
        beta = trial.suggest_float("beta", 0.1, 0.6)
        gamma = trial.suggest_float("gamma", 0.1, 0.5)
        tau_sim = trial.suggest_float("tau_sim", 0.25, 0.45)

        trial_weights = {
            "w_v": w_v,
            "w_a": w_a,
            "w_sigma": w_sigma,
            "w_s": w_s,
            "b": b,
            "tau_B": tau_B,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "tau_sim": tau_sim,
            "L_min": 80,
            "L_max": 400,
            "window_w": 5,
        }

        all_chunks = []
        global_cid = 0
        for item in doc_data:
            chunks = build_chunks_for_variant(
                variant="full_stp",
                units=item["units"],
                stp=item["stp"],
                embeddings=item["embs"],
                params=trial_weights,
            )
            for c in chunks:
                c.chunk_id = global_cid
                c.metadata["doc_id"] = item["doc_id"]
                global_cid += 1
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0.0

        store = STPVectorStore(
            collection_name=f"calib_trial_{trial.number}",
            embedding_dim=all_chunks[0].embedding.shape[0],
            use_memory=True,
        )
        store.index_chunks(all_chunks)
        retriever = DenseRetriever(vector_store=store, embedder=embedder)

        queries = []
        gold_chunk_ids_list = []
        for item in doc_data:
            doc_cids = {c.chunk_id for c in all_chunks if c.metadata.get("doc_id") == item["doc_id"]}
            for qa in item["qa_pairs"]:
                gold_sentences = qa.get("gold_sentences") or []
                matching_cids = set()
                for c in all_chunks:
                    if c.chunk_id in doc_cids:
                        if any(gs.lower() in c.text.lower() for gs in gold_sentences if gs):
                            matching_cids.add(c.chunk_id)
                if matching_cids:
                    queries.append(qa["question"])
                    gold_chunk_ids_list.append(matching_cids)

        if not queries:
            return 0.0

        retrieved_cids, _ = retriever.batch_retrieve(queries, top_k=5)
        r5 = compute_recall_at_k(retrieved_cids, gold_chunk_ids_list, k=5)
        return float(r5)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptuna Search Complete! Best dev Recall@5: {study.best_value:.4f}")
    best_params = study.best_params

    # Final Theta configuration
    final_weights = {
        "w_v": round(float(best_params["w_v"]), 4),
        "w_a": round(float(best_params["w_a"]), 4),
        "w_sigma": round(float(best_params["w_sigma"]), 4),
        "w_s": round(float(best_params["w_s"]), 4),
        "b": round(float(best_params["b"]), 4),
        "tau_B": round(float(best_params["tau_B"]), 4),
        "alpha": round(float(best_params["alpha"]), 4),
        "beta": round(float(best_params["beta"]), 4),
        "gamma": round(float(best_params["gamma"]), 4),
        "tau_sim": round(float(best_params["tau_sim"]), 4),
        "L_min": 80,
        "L_max": 400,
        "eta_heading": 1.0,
        "eta_list": 0.5,
        "eta_paragraph": 0.8,
        "eta_clause": 0.6,
        "H_min": 1,
        "H_max": 4,
        "lambda_1": 0.45,
        "lambda_2": 0.40,
        "lambda_3": 0.90,
        "mu_1": 0.50,
        "mu_2": 0.30,
        "mu_3": 0.50,
        "tau_R": 0.40,
        "window_w": 5,
    }

    calibrated_config = {
        "normalization": norm_stats,
        "structural_scaling": structural_scaling,
        "weights": final_weights,
    }

    # --- Stage C: Freeze winning config and compute SHA-256 ---
    out_p = Path(config_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.dump(calibrated_config, f, sort_keys=False)

    sha256_hash = compute_file_sha256(out_p)
    sha_p = out_p.with_suffix(".sha256")
    sha_p.write_text(sha256_hash, encoding="utf-8")

    print(f"\n[Stage C] Winning Theta frozen to: {out_p}")
    print(f"Configuration SHA-256 written to {sha_p}: {sha256_hash}")
    return calibrated_config, sha256_hash


def main():
    run_calibration()


if __name__ == "__main__":
    main()
