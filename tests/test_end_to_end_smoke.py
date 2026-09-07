"""End-to-end smoke test validating the full STP-RAG pipeline on a test document."""

import numpy as np
import pytest

from stp_rag.ingestion.structure_units import extract_structure_aware_units
from stp_rag.profile.structural_features import compute_structural_score
from stp_rag.profile.stp import compute_stp
from stp_rag.chunking.variants import build_chunks_for_variant
from stp_rag.context.selective_propagation import select_and_propagate_context
from stp_rag.indexing.qdrant_store import STPVectorStore
from stp_rag.eval.results_schema import ArmResult, write
from stp_rag.eval.retrieval_metrics import compute_recall_at_k, compute_mrr
from stp_rag.eval.answer_metrics import compute_qa_metrics


SAMPLE_DOC = """# Title: Semantic Transition Control in Long Documents
Dense passage retrieval relies on dual-encoder architectures that project queries into embedding space.
Standard retrieval designs apply fixed-size chunking across document boundaries without accounting for internal semantic shifts.
However, documents are not uniformly dense in informational content.
Certain passages transition rapidly across topical boundaries, while others remain stable across several paragraphs.

# 2. Kinematic Analogies
To capture these dynamics, we introduce kinematic analogies.
Semantic velocity represents the rate of semantic displacement per normalized unit length.
Semantic acceleration measures the rate of change of semantic velocity.
When semantic acceleration is positive, it signals an approaching or active topical shift.
Conversely, negative acceleration indicates that the narrative is stabilizing into a cohesive topic.

# 3. Context Propagation
In this condition, preceding definitions must be preserved to resolve anaphoric dependencies.
Therefore, selective context propagation links necessary predecessor passages without creating indexing bloat."""


def test_full_pipeline_smoke():
    # 1. Ingestion
    units = extract_structure_aware_units(SAMPLE_DOC, granularity="sentence")
    assert len(units) >= 8

    # 2. Embeddings (synthetic normalized vectors for smoke test)
    n = len(units)
    np.random.seed(42)
    raw_embs = np.random.randn(n, 32).astype(np.float32)
    embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

    # 3. Structural features & STP
    cum_toks = np.array([u.cumulative_tokens for u in units])
    s_scores, _, _ = compute_structural_score(units)
    stp = compute_stp(embs, cum_toks, s_scores, window_w=3)

    assert len(stp.v_hat) == n
    assert len(stp.a_hat) == n
    assert len(stp.sigma_hat) == n

    # 4. Chunking for Full STP + selective context
    chunks = build_chunks_for_variant(
        "full_stp_context",
        units=units,
        stp=stp,
        embeddings=embs,
        params={"L_min": 30, "L_max": 120, "tau_B": 0.4},
    )
    assert len(chunks) >= 2

    # 5. Selective context propagation
    chunks = select_and_propagate_context(chunks, stp)

    # 6. In-memory indexing via STPVectorStore
    store = STPVectorStore(collection_name="test_smoke_col", embedding_dim=32, use_memory=True)
    build_time = store.index_chunks(chunks)
    assert build_time >= 0.0

    # 7. Search & Retrieval
    query_vec = embs[0]
    retrieved_ids, latency_ms = store.search(query_vec, top_k=3)
    assert len(retrieved_ids) > 0

    # 8. Retrieval Metrics
    gold_chunk_ids = [{0}]
    recall5 = compute_recall_at_k([retrieved_ids], gold_chunk_ids, k=5)
    mrr = compute_mrr([retrieved_ids], gold_chunk_ids)

    # 9. QA Metrics
    preds = ["an approaching or active topical shift"]
    golds = ["an approaching or active topical shift"]
    em, f1 = compute_qa_metrics(preds, golds)
    assert em == 1.0
    assert f1 == 1.0

    # 10. Result Schema Creation
    arm_result = ArmResult(
        system="full_stp_context",
        seed=42,
        split="test",
        recall_at_1=1.0,
        recall_at_5=recall5,
        recall_at_10=1.0,
        mrr=mrr,
        em=em,
        f1=f1,
        chunks_per_doc=float(len(chunks)),
        tokens_per_chunk=float(sum(c.token_count for c in chunks) / len(chunks)),
        build_time_sec=build_time,
        retrieval_latency_ms=latency_ms,
    )
    assert arm_result.recall_at_5 == 1.0
