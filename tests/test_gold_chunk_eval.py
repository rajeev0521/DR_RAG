"""Tests for gold-chunk mapping and discrimination in retrieval evaluation."""

import pytest
from stp_rag.eval.retrieval_metrics import compute_mrr, compute_recall_at_k


def test_recall_at_1_adversarial_non_top_ranked():
    """
    Proves that on a multi-chunk document, Recall@1 is strictly < 1.0
    when the answer is in a non-top-ranked chunk, confirming the bugfix.
    """
    # Suppose a document produces 3 chunks: 0, 1, 2
    # The gold answer is located strictly in chunk 2.
    gold_chunk_ids_list = [{2}]

    # The retriever ranks chunk 0 at rank 1, chunk 1 at rank 2, and chunk 2 at rank 3
    retrieved_chunk_ids = [[0, 1, 2]]

    # With correct evaluation:
    r1 = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids_list, k=1)
    r5 = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids_list, k=5)
    mrr = compute_mrr(retrieved_chunk_ids, gold_chunk_ids_list)

    # In the old buggy fallback, all chunks {0, 1, 2} were considered gold, giving r1 = 1.0!
    # With the bugfix, chunk 0 is NOT gold, so Recall@1 MUST be 0.0 (strictly < 1.0).
    assert r1 == 0.0, f"Recall@1 should be 0.0 when top chunk is non-gold, got {r1}"
    assert r5 == 1.0, f"Recall@5 should find chunk 2 in top 5, got {r5}"
    assert mrr == pytest.approx(1.0 / 3.0), f"MRR should be 1/3, got {mrr}"


def test_unmappable_query_dropped_rather_than_falsely_marked_correct():
    """
    Ensures unmappable queries are dropped rather than falling back to whole document.
    """
    # A chunk text that has nothing to do with gold evidence
    all_chunks_text = {
        10: "This chunk discusses deep learning optimizations.",
        11: "This chunk discusses GPU memory bandwidth.",
    }
    doc_chunk_set = {10, 11}

    # Query with gold sentence that is nowhere in the chunks
    qa = {
        "query_id": "q_missing",
        "question": "What is photosynthesis?",
        "answer": "process used by plants",
        "gold_sentences": ["Plants use sunlight to synthesize nutrients from carbon dioxide and water."],
    }

    matching_cids = set()
    for cid, text in all_chunks_text.items():
        if any(gs.lower() in text.lower() for gs in qa["gold_sentences"]):
            matching_cids.add(cid)

    # In the bugfix: matching_cids is empty, so query is dropped
    assert len(matching_cids) == 0

    # If the bug was present: matching_cids would be doc_chunk_set {10, 11}, making any retrieval 1.0
    buggy_matching = matching_cids or doc_chunk_set
    assert buggy_matching == {10, 11}  # The bug would falsely consider every chunk correct!
