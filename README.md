# Adaptive Semantic-Velocity Chunking with Dynamic-Relevant RAG (DR-RAG)

## Overview

Retrieval-Augmented Generation (RAG) systems face fundamental limitations when processing long, structured documents such as legal contracts and academic papers. Standard fixed-size chunking disrupts semantic coherence, while single-stage retrieval fails to support multi-hop reasoning across weakly linked evidence.

This project introduces **Adaptive Semantic-Velocity Chunking with Dynamic-Relevant RAG (DR-RAG)**, a two-phase framework that addresses these challenges through context-aware chunking and a lightweight classifier-routed two-stage retrieval mechanism.

## Motivation

Current RAG pipelines exhibit critical structural weaknesses when applied to domain-specific documents:

1. **Semantic Fragmentation**: Fixed-size token windows split logical units (e.g., legal clauses, mathematical propositions) across multiple chunks.
2. **Context Decay**: Later chunks lose access to definitions, assumptions, or conditions established in earlier sections.
3. **Expensive Multi-Hop Retrieval**: Single-stage dense retrieval fails for multi-hop reasoning, while graph-based and iterative retrieval methods are computationally expensive.
4. **Lack of Chunking Metrics**: The literature lacks a formal, quantitative metric for evaluating chunk boundary quality.

## Framework Architecture

### Phase I: Context-Augmented Agentic Chunking (CAAC)

Phase I replaces static chunking with Small Language Model (SLM)-guided semantic boundaries.

* **Two-Pass Document Segmentation**: Documents are first segmented into sentences and embedded to compute semantic drift. Adaptive chunk refinement then merges or splits chunks based on their semantic drift values.
* **Semantic Velocity**: A novel metric quantifying the magnitude of meaning shift between consecutive chunks. It regulates adaptive chunk sizing and summary length.
* **Rolling Summaries**: Each chunk is enriched with a rolling summary of preceding context, ensuring preservation of both local meaning and global document flow.
* **Context Retention Score (CRS)**: A metric to evaluate whether a chunk remains semantically understandable in isolation.

### Phase II: Dynamic-Relevant RAG (DR-RAG)

Phase II defines the query-time pipeline using a classifier-routed two-stage retrieval mechanism.

* **Query Routing**: A lightweight classifier routes incoming queries as either single-hop or multi-hop based on embedding vector, syntactic complexity, conjunction frequency, and semantic entropy.
* **Anchor Retrieval**: Retrieves the top-k most semantically similar chunks using cosine similarity.
* **Evidence Mining**: For multi-hop queries, SLM-guided structured extraction extracts intermediate facts from anchor chunks to synthesize expanded queries and retrieve additional supporting evidence.
* **Context Assembly**: Anchor and evidence chunks are deduplicated and fed into the generation model to produce the final response.

## Evaluation and Ablation Study

To isolate the contribution of each component, the framework will be evaluated across four configurations:

* **A1**: No rolling summaries
* **A2**: Fixed-size chunking instead of CAAC
* **A3**: No query routing (all queries use two-stage)
* **A4**: No evidence mining stage

Evaluation metrics include Recall@K, Exact Match (EM), Hallucination Rate, Latency, and Context Retention Score (CRS), using the HotpotQA and CUAD benchmarks.

## Author

Rajeev Gupta  
Department of Computer Science and Engineering
Sharda University

Mohammad Sameer
Department of Computer Science and Engineering
Sharda University
