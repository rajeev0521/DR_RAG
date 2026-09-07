"""Full end-to-end experiment pipeline orchestrator."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import yaml

from ..chunking.segment import Chunk
from ..chunking.variants import build_chunks_for_variant
from ..context.selective_propagation import select_and_propagate_context
from ..eval.answer_metrics import compute_qa_metrics
from ..eval.efficiency_metrics import compute_efficiency_metrics
from ..eval.results_schema import ArmResult, write
from ..eval.retrieval_metrics import compute_mrr, compute_recall_at_k
from ..generation.answer import AnswerGenerator
from ..indexing.qdrant_store import STPVectorStore
from ..ingestion.structure_units import extract_structure_aware_units
from ..profile.embed import EmbeddingModelWrapper
from ..profile.stp import compute_stp, fit_normalization_stats
from ..profile.structural_features import compute_structural_score
from ..eval.ragas_eval import evaluate_ragas
from ..retrieval.stage1_dense import DenseRetriever
from ..tracking.mlflow_utils import MLflowTracker


def compute_file_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a configuration file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_yaml_config(filepath: str | Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def precompute_corpus_embeddings(
    corpus_path: str = "data/splits/dev.json",
    cache_path: str = "data/cache/embeddings.parquet",
    model_name: str = "BAAI/bge-m3",
) -> Path:
    """
    Precomputes unit embeddings for an entire corpus and caches to parquet.
    """
    import pandas as pd

    p_cache = Path(cache_path)
    p_cache.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    embedder = EmbeddingModelWrapper(model_name=model_name)
    records = []

    for doc in docs:
        doc_id = doc["doc_id"]
        text = doc["text"]
        units = extract_structure_aware_units(text)
        unit_texts = [u.text for u in units]
        embs = embedder.embed_texts(unit_texts)

        for u_idx, (u, emb) in enumerate(zip(units, embs)):
            records.append({
                "doc_id": doc_id,
                "unit_id": u.unit_id,
                "text": u.text,
                "token_count": u.token_count,
                "cumulative_tokens": u.cumulative_tokens,
                "follows_heading": u.follows_heading,
                "follows_list_item": u.follows_list_item,
                "is_paragraph_start": u.is_paragraph_start,
                "follows_clause_marker": u.follows_clause_marker,
                "embedding": emb.tolist(),
            })

    df = pd.DataFrame(records)
    df.to_parquet(p_cache)
    return p_cache


def run_ablation_arm(
    variant: str,
    seed: int = 42,
    split: str = "test",
    data_path: Optional[str] = None,
    config_path: str = "configs/stp_params.yaml",
    retrieval_config_path: str = "configs/retrieval.yaml",
    eval_config_path: str = "configs/eval.yaml",
    results_dir: str = "results/",
    use_in_memory_store: bool = True,
) -> ArmResult:
    """
    Executes a single ablation arm run across a dataset split with seed control.
    """
    np.random.seed(seed)
    # Assert configuration checksum integrity if frozen checksum exists (Task 6)
    cfg_file = Path(config_path)
    sha_file = cfg_file.with_suffix(".sha256")
    if sha_file.exists():
        expected_sha = sha_file.read_text(encoding="utf-8").strip()
        actual_sha = compute_file_sha256(cfg_file)
        if actual_sha != expected_sha:
            raise ValueError(
                f"STP config checksum mismatch for {config_path}! "
                f"Expected {expected_sha}, got {actual_sha}. Config has been modified since freeze."
            )

    params_cfg = load_yaml_config(config_path)
    retrieval_cfg = load_yaml_config(retrieval_config_path)
    eval_cfg = load_yaml_config(eval_config_path) if Path(eval_config_path).exists() else {}

    # 1. Load Data
    split_file = data_path or f"data/splits/{split}.json"
    with open(split_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    norm_stats = params_cfg.get("normalization")
    weights = params_cfg.get("weights", {})
    structural_scaling = params_cfg.get("structural_scaling", {})

    # 2. Ingest, embed, and segment
    embed_model_name = retrieval_cfg.get("embedding_model", "BAAI/bge-m3")
    embedder = EmbeddingModelWrapper(model_name=embed_model_name)

    all_chunks: List[Chunk] = []
    chunks_per_doc: List[List[Chunk]] = []
    global_chunk_id = 0

    for doc in documents:
        units = extract_structure_aware_units(doc["text"])
        unit_texts = [u.text for u in units]

        # Use normalized embeddings
        embs = embedder.embed_texts(unit_texts)
        cum_tokens = np.array([u.cumulative_tokens for u in units], dtype=np.int32)

        # Structural score
        s_score, _, _ = compute_structural_score(
            units,
            min_val=structural_scaling.get("min_val"),
            max_val=structural_scaling.get("max_val"),
        )

        # STP computation using frozen dev normalization stats
        stp = compute_stp(
            embeddings=embs,
            cumulative_tokens=cum_tokens,
            structural_score=s_score,
            norm_stats=norm_stats,
            window_w=int(weights.get("window_w", 5)),
        )

        # Build chunks for this arm's variant
        doc_chunks = build_chunks_for_variant(
            variant=variant,
            units=units,
            stp=stp,
            embeddings=embs,
            params=weights,
        )

        # Re-assign continuous global chunk IDs
        for c in doc_chunks:
            c.chunk_id = global_chunk_id
            c.metadata["doc_id"] = doc["doc_id"]
            global_chunk_id += 1

        # Selective context propagation (if variant is full_stp_context)
        if variant == "full_stp_context":
            doc_chunks = select_and_propagate_context(
                doc_chunks,
                stp=stp,
                horizon_params=weights,
                ranking_params=weights,
            )

        all_chunks.extend(doc_chunks)
        chunks_per_doc.append(doc_chunks)

    # 3. Vector Indexing into dedicated collection
    collection_name = f"col_{variant}__seed{seed}__{split}"
    dim = all_chunks[0].embedding.shape[0] if (all_chunks and all_chunks[0].embedding is not None) else 1024
    store = STPVectorStore(
        collection_name=collection_name,
        embedding_dim=dim,
        url=retrieval_cfg.get("vector_store", {}).get("url"),
        use_memory=use_in_memory_store,
    )
    build_time = store.index_chunks(all_chunks)

    # 4. Dense Retrieval and Generation
    retriever = DenseRetriever(vector_store=store, embedder=embedder)
    generator = AnswerGenerator(
        model_name=retrieval_cfg.get("generation", {}).get("model_name", "phi3:mini"),
        ollama_url=retrieval_cfg.get("generation", {}).get("ollama_url", "http://localhost:11434/api/generate"),
    )

    queries = []
    gold_answers = []
    gold_chunk_ids_list = []

    for doc in documents:
        doc_id = doc["doc_id"]
        # Find chunks belonging to this doc
        doc_chunk_set = {c.chunk_id for c in all_chunks if c.metadata.get("doc_id") == doc_id}

        for qa in doc.get("qa_pairs", []):
            gold_sentences = qa.get("gold_sentences") or []
            supp_facts = qa.get("supporting_facts") or []
            keywords = qa.get("gold_chunk_keywords") or []

            matching_cids = set()
            for c in all_chunks:
                if c.chunk_id in doc_chunk_set:
                    c_text_lower = c.text.lower()
                    # 1. Match on exact gold sentence / evidence text (or chunk contained inside long gold evidence paragraph)
                    if any(
                        (gs.lower() in c_text_lower or (len(c_text_lower) >= 30 and c_text_lower in gs.lower()))
                        for gs in gold_sentences if gs
                    ):
                        matching_cids.add(c.chunk_id)
                    # 2. Match on supporting facts if string-based
                    elif any(
                        isinstance(sf, str) and (sf.lower() in c_text_lower or (len(c_text_lower) >= 30 and c_text_lower in sf.lower()))
                        for sf in supp_facts if sf
                    ):
                        matching_cids.add(c.chunk_id)
                    # 3. Fallback to keywords only if present in legacy fixtures
                    elif keywords and any(kw.lower() in c_text_lower for kw in keywords if kw):
                        matching_cids.add(c.chunk_id)

            if not matching_cids:
                # Do NOT fall back to whole doc; drop unmappable query to avoid false 1.0 Recall/MRR
                continue

            queries.append(qa["question"])
            gold_answers.append(qa["answer"])
            gold_chunk_ids_list.append(matching_cids)

    # Execute batch retrieval
    retrieved_chunk_ids, query_latencies = retriever.batch_retrieve(queries, top_k=10)

    # 5. Answer Generation
    chunk_map = {c.chunk_id: c for c in all_chunks}
    max_eval_q = eval_cfg.get("generation", {}).get("max_eval_queries")
    eval_query_indices = list(range(len(queries)))
    if max_eval_q and len(queries) > max_eval_q:
        eval_query_indices = eval_query_indices[:max_eval_q]

    predicted_answers = []
    evaluated_gold_answers = []
    for q_idx in eval_query_indices:
        q_text = queries[q_idx]
        top_cids = retrieved_chunk_ids[q_idx][:3]
        retrieved_contexts = [chunk_map[cid].augmented_text for cid in top_cids if cid in chunk_map]
        ans, _ = generator.generate(q_text, retrieved_contexts)
        predicted_answers.append(ans)
        evaluated_gold_answers.append(gold_answers[q_idx])

    # 6. Evaluation Metrics
    r1 = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids_list, k=1)
    r5 = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids_list, k=5)
    r10 = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids_list, k=10)
    mrr = compute_mrr(retrieved_chunk_ids, gold_chunk_ids_list)

    # Compute EM/F1 against generated answers
    em, f1 = compute_qa_metrics(predicted_answers, evaluated_gold_answers)
    eff = compute_efficiency_metrics(chunks_per_doc, build_time, query_latencies)

    # Compute RAGAS metrics if enabled (or populate explicit null sentinels)
    ragas_scores = {
        "ragas_faithfulness": None,
        "ragas_answer_relevancy": None,
        "ragas_context_precision": None,
        "ragas_context_recall": None,
    }
    eval_metrics_cfg = eval_cfg.get("metrics", {})
    if eval_metrics_cfg.get("compute_ragas", False) and queries:
        query_contexts = []
        for q_idx in range(len(queries)):
            top_cids = retrieved_chunk_ids[q_idx][:3]
            query_contexts.append([chunk_map[cid].augmented_text for cid in top_cids if cid in chunk_map])

        ragas_limit = eval_metrics_cfg.get("ragas_sample_size", 10)
        ragas_scores = evaluate_ragas(
            questions=queries[:ragas_limit],
            contexts=query_contexts[:ragas_limit],
            answers=predicted_answers[:ragas_limit],
            ground_truths=gold_answers[:ragas_limit],
            judge_model=eval_metrics_cfg.get("judge_model", "phi3:mini"),
            embedding_model=embed_model_name,
            ollama_url=retrieval_cfg.get("generation", {}).get("ollama_url", "http://localhost:11434/api/generate").replace("/api/generate", ""),
        )

    arm_result = ArmResult(
        system=variant,
        seed=seed,
        split=split,
        recall_at_1=float(r1),
        recall_at_5=float(r5),
        recall_at_10=float(r10),
        mrr=float(mrr),
        em=float(em),
        f1=float(f1),
        chunks_per_doc=float(eff["chunks_per_doc"]),
        tokens_per_chunk=float(eff["tokens_per_chunk"]),
        build_time_sec=float(eff["build_time_sec"]),
        retrieval_latency_ms=float(eff["retrieval_latency_ms"]),
        ragas_faithfulness=ragas_scores.get("ragas_faithfulness"),
        ragas_answer_relevancy=ragas_scores.get("ragas_answer_relevancy"),
        ragas_context_precision=ragas_scores.get("ragas_context_precision"),
        ragas_context_recall=ragas_scores.get("ragas_context_recall"),
        embedding_model_version=embed_model_name,
        corpus_version="1.0",
    )

    # 7. Start MLflow run first, capture real run ID, log result, and persist JSON (Task 5)
    tracker = MLflowTracker()
    with tracker.run(run_name=f"{variant}_seed{seed}_{split}") as run_id:
        arm_result.mlflow_run_id = str(run_id)
        tracker.log_arm_result(arm_result)
        write(arm_result, results_dir=results_dir)

    return arm_result
