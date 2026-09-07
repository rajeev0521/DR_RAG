"""Dataset preparation and dev/test stratification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def create_benchmark_sample_data(
    dev_path: str = "data/splits/dev.json",
    test_path: str = "data/splits/test.json",
):
    """
    Creates guaranteed, reproducible benchmark splits for long-document RAG evaluation.
    Each split contains documents with distinct topical sections, definitions,
    and corresponding multi-hop and single-hop question-answer pairs.
    """
    dev_docs = [
        {
            "doc_id": "doc_dev_01",
            "title": "Kinematic Representation in Dense Retrieval Systems",
            "text": """# 1. Introduction to Retrieval Kinematics
Dense passage retrieval relies on dual-encoder architectures that project queries and candidate text passages into a continuous embedding space.
Standard retrieval designs apply fixed-size chunking across document boundaries without accounting for internal semantic shifts.
However, documents are not uniformly dense in informational content.
Certain passages transition rapidly across topical boundaries, while others remain stable across several paragraphs.

# 2. Velocity and Acceleration Cues
To capture these dynamics, we introduce kinematic analogies.
Semantic velocity represents the rate of semantic displacement per normalized unit length.
Semantic acceleration measures the rate of change of semantic velocity.
When semantic acceleration is positive, it signals an approaching or active topical shift.
Conversely, negative acceleration indicates that the narrative is stabilizing into a cohesive topic.

# 3. Reference and Context Resolution
In this condition, preceding definitions must be preserved to resolve anaphoric dependencies.
For example, abbreviations such as STP (Semantic Transition Profile) require their antecedent definitions.
Therefore, selective context propagation links necessary predecessor passages without creating indexing bloat.""",
            "qa_pairs": [
                {
                    "query_id": "q_dev_01",
                    "question": "What does positive semantic acceleration indicate in a document?",
                    "answer": "an approaching or active topical shift",
                    "gold_chunk_keywords": ["acceleration", "topical shift"],
                    "query_type": "single_hop",
                },
                {
                    "query_id": "q_dev_02",
                    "question": "Why is selective context propagation needed when anaphoric dependencies like STP occur?",
                    "answer": "to link necessary predecessor passages without creating indexing bloat",
                    "gold_chunk_keywords": ["selective context", "indexing bloat"],
                    "query_type": "multi_hop",
                },
            ],
        },
        {
            "doc_id": "doc_dev_02",
            "title": "Long-Document Boundary Calibration in Legal Corpora",
            "text": """# Article I: Scope of Obligations
The recipient shall preserve all proprietary information disclosed by the disclosing party under strict confidentiality.
Proprietary information includes source code, financial algorithms, customer lists, and developmental roadmaps.
However, proprietary information does not include information that is publicly known or independently developed without reference to the disclosure.

# Article II: Remedies and Exceptions
Provided that an unauthorized disclosure occurs, the aggrieved party is entitled to immediate injunctive relief.
Notwithstanding any contrary provision herein, liability for gross negligence shall not be limited.
Therefore, neither party may disclaim damages arising from intentional misconduct or willful breach of Article I.""",
            "qa_pairs": [
                {
                    "query_id": "q_dev_03",
                    "question": "Under what condition is the aggrieved party entitled to immediate injunctive relief?",
                    "answer": "Provided that an unauthorized disclosure occurs",
                    "gold_chunk_keywords": ["injunctive relief", "unauthorized disclosure"],
                    "query_type": "single_hop",
                }
            ],
        },
    ]

    test_docs = [
        {
            "doc_id": "doc_test_01",
            "title": "Adaptive Chunk Granularity and Index Efficiency",
            "text": """# 1. Background on Segmentation Trade-offs
Fixed-window chunking imposes a single token budget across heterogeneous document regions.
When chunks are too large, irrelevant claims pollute the representation and lower retrieval precision.
When chunks are too small, core definitions are severed from their operational conditions.
This trade-off governs the effectiveness of long-context retrieval-augmented generation.

# 2. Semantic Volatility and Granularity Policies
Semantic volatility quantifies the local standard deviation of semantic velocity across a trailing window.
High volatility indicates frequent, erratic topic changes such as enumerations, exceptions, or technical definitions.
In volatile regions, the target chunk budget is dynamically scaled down toward the minimum chunk length L_min.
In stable regions, the target budget expands toward L_max to minimize index overhead and retain holistic context.

# 3. Empirical Implications on Latency and Quality
Therefore, volatility-guided granularity creates fewer total chunks than uniform micro-chunking while preserving fine-grained boundaries.
As a result, retrieval latency remains competitive while Mean Reciprocal Rank improves.""",
            "qa_pairs": [
                {
                    "query_id": "q_test_01",
                    "question": "How is the target chunk budget adjusted in highly volatile document regions?",
                    "answer": "dynamically scaled down toward the minimum chunk length L_min",
                    "gold_chunk_keywords": ["volatility", "L_min"],
                    "query_type": "single_hop",
                },
                {
                    "query_id": "q_test_02",
                    "question": "What are the two failure modes when chunks are either too large or too small?",
                    "answer": "irrelevant claims pollute the representation and core definitions are severed from operational conditions",
                    "gold_chunk_keywords": ["too large", "too small", "pollute"],
                    "query_type": "multi_hop",
                },
            ],
        },
        {
            "doc_id": "doc_test_02",
            "title": "Context Horizon Scaling and Predecessor Ranking",
            "text": """# Section A: Context Selection Mechanics
Attaching a fixed historical window to every chunk inflates the index and degrades vector search specificity.
STP-RAG computes an adaptive context horizon H_i determined by semantic volatility, positive acceleration, and reference cues.
The horizon specifies the candidate predecessor pool size, bounded between H_min and H_max.

# Section B: Ranking and Filtering Candidates
Candidate predecessors within the horizon are scored using composite relevance.
The relevance function combines embedding cosine similarity, entity dependency overlap, and reference resolution matching.
Only candidates surpassing the calibrated threshold tau_R are attached to the indexed chunk.
Consequently, only contextually vital predecessors are preserved, ensuring high faithfulness during generation.""",
            "qa_pairs": [
                {
                    "query_id": "q_test_03",
                    "question": "What three signals determine the adaptive context horizon H_i?",
                    "answer": "semantic volatility, positive acceleration, and reference cues",
                    "gold_chunk_keywords": ["horizon", "volatility", "acceleration"],
                    "query_type": "single_hop",
                },
                {
                    "query_id": "q_test_04",
                    "question": "Which candidate predecessors are attached to the indexed chunk?",
                    "answer": "Only candidates surpassing the calibrated threshold tau_R",
                    "gold_chunk_keywords": ["tau_R", "attached"],
                    "query_type": "single_hop",
                },
            ],
        },
    ]

    p_dev = Path(dev_path)
    p_dev.parent.mkdir(parents=True, exist_ok=True)
    with open(p_dev, "w", encoding="utf-8") as f:
        json.dump(dev_docs, f, indent=2)

    p_test = Path(test_path)
    p_test.parent.mkdir(parents=True, exist_ok=True)
    with open(p_test, "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2)

    return len(dev_docs), len(test_docs)


if __name__ == "__main__":
    n_dev, n_test = create_benchmark_sample_data()
    print(f"Stratified benchmark data created: {n_dev} dev docs, {n_test} test docs.")
