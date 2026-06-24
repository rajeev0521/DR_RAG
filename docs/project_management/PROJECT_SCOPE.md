# Project Scope: Adaptive Semantic-Velocity Chunking with Dynamic-Relevant RAG

## 1. Problem Statement

Retrieval-Augmented Generation (RAG) systems are increasingly adopted for question answering over large document corpora. However, when applied to long, structured documents — such as legal contracts, academic papers, technical reports, and policy documents — conventional RAG pipelines exhibit critical limitations:

1. **Semantic Fragmentation**: Fixed-size token windows break logical units (legal clauses, mathematical derivations, argument chains) across multiple chunks, destroying the coherence necessary for accurate retrieval and generation.

2. **Context Decay**: Chunks extracted from later sections of a document lose access to definitions, assumptions, and conditions established earlier. This results in semantically incomplete retrieval units that produce inaccurate or hallucinated answers.

3. **Single-Stage Retrieval Failure**: Standard dense retrieval computes similarity between the raw query and stored chunk embeddings in a single pass. For multi-hop queries — where the answer depends on synthesizing information across semantically distant chunks — this approach fails because supporting evidence may have low direct similarity to the original query.

4. **Absence of Chunking Quality Metrics**: The RAG literature lacks a formal, quantitative metric for evaluating whether chunk boundaries preserve semantic coherence.

These limitations motivate the need for an adaptive chunking strategy that respects document structure and a multi-stage retrieval mechanism that can discover weakly linked supporting evidence.

---

## 2. Research Objectives

The project pursues five primary research objectives:

| ID | Objective | Description |
|---|---|---|
| **RO1** | Adaptive Semantic Chunking | Design and implement a chunking strategy (CAAC) that uses SLM-guided boundary detection, semantic velocity measurement, and adaptive refinement to produce semantically coherent chunks from structured documents. |
| **RO2** | Context Preservation | Develop a rolling summary mechanism that augments each chunk with compressed context from preceding chunks, mitigating context decay in long documents. |
| **RO3** | Query-Adaptive Retrieval | Build a two-stage retrieval pipeline (DR-RAG) that routes queries based on complexity, performs anchor retrieval, extracts intermediate evidence, and mines additional supporting chunks for multi-hop reasoning. |
| **RO4** | Quantitative Evaluation | Establish a rigorous evaluation framework comparing the proposed system against a fixed-chunk baseline RAG using retrieval quality metrics (Recall@K, MRR), generation quality metrics (Exact Match, F1, RAGAS), and system efficiency metrics (latency, hallucination rate). |
| **RO5** | Component Contribution Analysis | Conduct an ablation study isolating the contribution of each major component (rolling summaries, adaptive chunking, query routing, evidence mining) to overall system performance. |

---

## 3. Expected Contributions

This project makes the following contributions to the RAG research domain:

1. **Context-Augmented Agentic Chunking (CAAC)**: A novel chunking pipeline that combines SLM-guided semantic boundary detection with semantic velocity measurement and adaptive chunk refinement (merge/retain/split). Unlike existing approaches (RAPTOR, GraphRAG), CAAC operates without expensive graph construction or hierarchical tree building.

2. **Semantic Velocity Metric**: A quantitative measure of meaning shift between consecutive text segments, computed as the embedding distance between adjacent sentence groups. Semantic velocity drives three adaptive mechanisms: chunk boundary decisions, chunk size adjustment, and summary token budget scaling.

3. **Context Retention Score (CRS)**: A proposed metric for evaluating whether a chunk remains semantically understandable in isolation, addressing the gap in chunking quality evaluation.

4. **Dynamic-Relevant RAG (DR-RAG)**: A classifier-routed two-stage retrieval pipeline that bridges multi-hop reasoning gaps by mining additional evidence from intermediate facts extracted from anchor chunks. This achieves improved multi-hop recall without the latency overhead of iterative LLM reasoning loops used in agentic RAG systems.

5. **Comprehensive Ablation Study**: Systematic evaluation of component-level contributions across four ablation configurations, providing evidence for the necessity of each architectural component.

---

## 4. Input Specification

### 4.1 Supported Document Types

| Document Type | Structural Characteristics | Example Sources |
|---|---|---|
| **Legal Contracts** | Hierarchical clauses, cross-references, defined terms, numbered sections | CUAD dataset, sample NDAs, service agreements |
| **Academic Papers** | Abstract → Introduction → Methodology → Results → Conclusion; citations, equations, figures | ArXiv papers, conference proceedings |
| **Technical Reports** | Numbered sections, tables, specifications, appendices | Engineering reports, standards documents |
| **Policy Documents** | Regulatory language, hierarchical sections, definitions, compliance requirements | Government policy PDFs, institutional guidelines |

### 4.2 Supported Input Formats

- **PDF** (text-based; scanned/image PDFs are out of scope)
- **Plain text** (.txt)
- **Markdown** (.md)

### 4.3 Language

- **English only** in the current implementation scope.

---

## 5. Query Specification

### 5.1 Single-Hop Queries

Queries answerable from a single contiguous chunk or a small set of highly similar chunks.

**Examples:**
- "What is the termination clause in this contract?"
- "What embedding model did the authors use?"
- "What is the penalty for late delivery?"

### 5.2 Multi-Hop Queries

Queries requiring synthesis of information from multiple semantically distant chunks.

**Examples:**
- "Does the indemnification clause cover scenarios described in the force majeure section?"
- "How does the methodology address the limitations identified in the literature review?"
- "What evidence supports the claim that the proposed approach outperforms the baseline, and under what conditions does it fail?"

---

## 6. Output Specification

The system produces structured responses containing:

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | The generated natural language answer grounded in retrieved evidence. |
| `source_chunks` | `list[ChunkReference]` | List of chunk citations used to produce the answer, each containing chunk ID, document ID, page number, section title, and relevance score. |
| `query_type` | `string` | Classification result: `"single_hop"` or `"multi_hop"`. |
| `retrieval_stages` | `object` | For multi-hop queries: anchor chunk IDs, extracted intermediate facts, mining query text, and evidence chunk IDs. |
| `confidence` | `float` | Estimated answer confidence based on retrieval score aggregation. |

---

## 7. Success Criteria

The proposed DR-RAG system will be evaluated against a fixed-chunk baseline RAG. The following measurable targets define success:

| Metric | Baseline Target | DR-RAG Target | Improvement |
|---|---|---|---|
| **Recall@5** | Establish baseline | ≥ 10% improvement over baseline | Relative |
| **MRR** | Establish baseline | ≥ 10% improvement over baseline | Relative |
| **RAGAS Faithfulness** | Establish baseline | ≥ 0.05 absolute improvement | Absolute |
| **RAGAS Context Precision** | Establish baseline | ≥ 0.05 absolute improvement | Absolute |
| **Hallucination Rate** | Establish baseline | ≥ 15% reduction over baseline | Relative |
| **Latency** | Establish baseline | ≤ 2x baseline latency for multi-hop | Bounded overhead |

> **Note:** Exact baseline values will be established in Week 4 (Baseline RAG Pipeline). Success is defined in terms of relative improvement over the measured baseline.

---

## 8. Scope Timeline Mapping

The project scope maps to the 20-week implementation plan as follows:

| Phase | Weeks | Scope Coverage |
|---|---|---|
| Foundation | 1–4 | Literature study, scope finalization, environment setup, baseline RAG |
| CAAC Pipeline | 5–10 | Document preprocessing, semantic velocity, boundary detection, chunk refinement, rolling summaries, token budget scaling |
| DR-RAG Pipeline | 11–16 | Vector indexing, query routing, anchor retrieval, evidence extraction, mining queries, evidence mining |
| Integration & Evaluation | 17–20 | Answer generation, API, evaluation, ablation study, final report |
