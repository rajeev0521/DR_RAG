# Project Plan: Adaptive Semantic-Velocity Chunking with Dynamic-Relevant RAG

## Project Title

**Adaptive Semantic-Velocity Chunking with Dynamic-Relevant Retrieval-Augmented Generation (DR-RAG)**

## Project Overview

This project proposes an improved Retrieval-Augmented Generation (RAG) framework for long, structured documents such as legal contracts, academic papers, technical reports, and policy documents. Traditional RAG pipelines commonly use fixed-size chunking and single-stage vector retrieval. These approaches often fail when the answer requires context from multiple sections of a document or when an important clause depends on definitions or assumptions introduced earlier.

The proposed system combines two main ideas:

1. **Context-Augmented Agentic Chunking (CAAC)**: an intelligent chunking method that uses semantic boundaries, document structure, rolling summaries, and semantic velocity to create context-aware chunks.
2. **Dynamic-Relevant RAG (DR-RAG)**: a two-stage retrieval pipeline that first retrieves anchor chunks and then mines additional supporting evidence for complex multi-hop queries.

The goal is to improve retrieval recall, context preservation, multi-hop reasoning, and hallucination resistance compared with conventional fixed-chunk RAG systems.

## Finalized Technology Stack

| Layer | Tool | Purpose | Justification |
|---|---|---|---|
| **Embeddings** | **BGE-M3** | Converts chunks and queries into dense vector representations. | BGE-M3 is strong for semantic retrieval, supports long and variable-sized text, and is suitable for academic and legal documents. |
| **Vector Database** | **Qdrant** | Stores chunk embeddings and metadata for similarity search. | Qdrant provides fast vector search, HNSW indexing, metadata filtering, Docker support, and clean APIs, making it suitable for research and prototype deployment. |
| **Boundary SLM** | **Phi-3 Mini** | Detects semantic boundaries, generates rolling summaries, and assists chunk refinement. | Phi-3 Mini is lightweight, can run locally, and is efficient for boundary detection and summarization tasks without high API cost. |
| **Query Router** | **DistilBERT** | Classifies user queries as single-hop or multi-hop. | DistilBERT is faster and smaller than BERT while retaining strong classification performance, making it suitable for low-latency query routing. |
| **Retrieval Framework** | **LlamaIndex** | Orchestrates document ingestion, indexing, retrieval, and custom RAG workflows. | LlamaIndex is document-centric and supports flexible retriever customization, metadata-aware retrieval, and vector database integrations. |
| **NLP Processing** | **spaCy** | Performs sentence splitting, tokenization, dependency parsing, and linguistic feature extraction. | spaCy is fast, reliable, and useful for preprocessing, sentence segmentation, and query complexity feature extraction. |
| **Evidence Extraction** | **AllenNLP OpenIE** | Extracts subject-relation-object triples and propositions from retrieved anchor chunks. | OpenIE helps convert retrieved text into structured intermediate facts, which are useful for second-stage evidence mining. |
| **Backend** | **FastAPI** | Provides APIs for document ingestion, retrieval, generation, and evaluation. | FastAPI is lightweight, fast, easy to document, and suitable for serving ML/NLP pipelines. |
| **Evaluation** | **RAGAS** | Evaluates RAG outputs using faithfulness, answer relevancy, context precision, and context recall. | RAGAS provides RAG-specific evaluation metrics and supports systematic comparison between baseline and proposed methods. |
| **Experiment Tracking** | **MLflow** | Tracks experiments, metrics, parameters, models, and ablation results. | MLflow makes it easier to compare chunking strategies, retrieval configurations, model versions, and evaluation runs. |
| **Training** | **PyTorch + HuggingFace** | Fine-tunes or trains query router and supporting transformer models. | PyTorch and HuggingFace provide standard tools for transformer model training, dataset handling, and evaluation. |
| **Deployment** | **Docker** | Containerizes the backend, vector database, and model-serving components. | Docker ensures reproducible deployment and simplifies running Qdrant, FastAPI, and local model services across environments. |

## Week-Wise Project Plan

### Week 1: Literature Study and Problem Understanding

**Tasks**

- Study the fundamentals of Retrieval-Augmented Generation.
- Review common RAG limitations such as hallucination, context fragmentation, and poor multi-hop retrieval.
- Read about fixed-size chunking, semantic chunking, RAPTOR, GraphRAG, DPR, and multi-stage retrieval.
- Identify why legal documents and academic papers are difficult for traditional RAG systems.

**Outcome**

- Clear understanding of the problem domain.
- Initial notes on research gaps and motivation.
- Basic comparison between traditional RAG and the proposed DR-RAG idea.

### Week 2: Scope Finalization and Architecture Design

**Tasks**

- Finalize the project scope, objectives, and expected contributions.
- Define the two major system phases: CAAC and DR-RAG.
- Prepare high-level architecture diagrams.
- Identify input document types, expected query types, and output format.

**Outcome**

- Finalized project objective.
- High-level architecture diagram.
- Phase-wise design for indexing and retrieval.
- Clear project boundaries for implementation.

### Week 3: Environment Setup

**Tasks**

- Set up Python environment and project repository.
- Install core libraries: LlamaIndex, Qdrant client, HuggingFace Transformers, PyTorch, spaCy, RAGAS, FastAPI, and MLflow.
- Set up Docker for Qdrant.
- Prepare sample datasets and test documents.

**Outcome**

- Working development environment.
- Qdrant running locally.
- Basic project folder structure.
- Initial test scripts for loading documents.

### Week 4: Baseline RAG Pipeline

**Tasks**

- Implement a standard RAG baseline using fixed-size chunking.
- Generate embeddings using BGE-M3.
- Store baseline chunks in Qdrant.
- Implement single-stage top-k retrieval.
- Generate answers using retrieved chunks.

**Outcome**

- Functional baseline RAG pipeline.
- Baseline retrieval and answer generation results.
- Reference system for later comparison.

### Week 5: Document Preprocessing Pipeline

**Tasks**

- Implement PDF and text extraction.
- Clean extracted text by removing noise, repeated headers, footers, and formatting artifacts.
- Use spaCy for sentence splitting and tokenization.
- Extract basic metadata such as document name, page number, section title, and paragraph position.

**Outcome**

- Clean document preprocessing module.
- Structured document representation ready for adaptive chunking.
- Metadata format finalized.

### Week 6: Semantic Velocity Calculation

**Tasks**

- Embed individual sentences using BGE-M3.
- Compute semantic drift between adjacent sentence embeddings.
- Define semantic velocity as embedding distance between neighboring segments.
- Visualize or log low, medium, and high semantic velocity regions.

**Outcome**

- Semantic velocity calculation module.
- Initial threshold values for low, moderate, and high topic shifts.
- Evidence that semantic velocity can identify topic transitions.

### Week 7: SLM-Guided Boundary Detection

**Tasks**

- Use Phi-3 Mini to identify semantic chunk boundaries.
- Create prompt templates for academic documents, legal documents, and general structured text.
- Combine SLM boundary suggestions with heading, paragraph, and semantic velocity signals.
- Test boundary detection on sample documents.

**Outcome**

- First version of the boundary detection module.
- Prompt templates for different document types.
- Initial semantic chunks generated from real documents.

### Week 8: Adaptive Chunk Refinement

**Tasks**

- Implement logic to merge chunks with low semantic velocity.
- Retain chunks with moderate semantic velocity.
- Split chunks around high semantic velocity transitions.
- Ensure each chunk remains within an acceptable token range.

**Outcome**

- Refined semantic chunking module.
- Chunks that are more coherent than fixed-size windows.
- Initial comparison between fixed-size chunks and semantic chunks.

### Week 9: Rolling Summary Generation

**Tasks**

- Use Phi-3 Mini to generate summaries of the previous three chunks.
- Prepend the rolling summary to the current chunk before embedding.
- Store both raw chunk and augmented chunk.
- Test whether augmented chunks preserve earlier context.

**Outcome**

- Context-augmented chunk construction completed.
- Rolling summary mechanism implemented.
- Improved chunk interpretability for later sections of long documents.

### Week 10: Token Budget Scaling

**Tasks**

- Implement summary token budget scaling based on semantic velocity.
- Use shorter summaries for low-drift transitions.
- Use longer summaries for high-drift transitions.
- Tune minimum and maximum summary lengths.

**Outcome**

- Final CAAC chunking pipeline.
- Adaptive summary lengths based on topic shift intensity.
- Reduced redundancy while preserving context.

### Week 11: Vector Indexing with Metadata

**Tasks**

- Store augmented chunks in Qdrant.
- Add metadata fields such as document ID, chunk ID, page number, section title, semantic velocity score, and document type.
- Implement metadata-aware retrieval.
- Validate indexing and search performance.

**Outcome**

- Complete vector index for CAAC chunks.
- Searchable chunk store with useful metadata.
- Foundation ready for DR-RAG retrieval.

### Week 12: Query Complexity Router

**Tasks**

- Build a query router using DistilBERT.
- Classify queries into single-hop and multi-hop categories.
- Train or fine-tune using HotpotQA, Natural Questions, and manually labeled examples.
- Add fallback logic for low-confidence predictions.

**Outcome**

- Functional query complexity classifier.
- Query routing integrated into retrieval pipeline.
- Simple queries routed to standard retrieval and complex queries routed to DR-RAG.

### Week 13: Stage 1 Anchor Retrieval

**Tasks**

- Implement Stage 1 retrieval using the original query.
- Retrieve top-k semantically similar chunks from Qdrant.
- Return anchor chunks with relevance scores and metadata.
- Tune top-k value based on recall and latency.

**Outcome**

- Anchor retrieval module completed.
- Retrieved chunks available for both simple answer generation and multi-hop evidence mining.

### Week 14: Intermediate Evidence Extraction

**Tasks**

- Use AllenNLP OpenIE to extract triples from anchor chunks.
- Extract entities, relationships, claims, and key propositions.
- Optionally use Phi-3 Mini to clean or refine noisy OpenIE outputs.
- Convert extracted facts into a structured intermediate representation.

**Outcome**

- Intermediate fact extraction module completed.
- Anchor chunks transformed into structured evidence.
- Facts ready for mining query generation.

### Week 15: Mining Query Generation

**Tasks**

- Combine the original query with extracted intermediate facts.
- Generate an expanded mining query.
- Ensure the mining query targets missing or weakly linked evidence.
- Test mining queries on multi-hop examples.

**Outcome**

- Mining query synthesis module completed.
- Expanded queries that improve the chance of retrieving supporting evidence.

### Week 16: Stage 2 Evidence Mining

**Tasks**

- Retrieve additional chunks using the mining query.
- Remove duplicates already present in anchor results.
- Rank and filter mined chunks.
- Combine anchor and evidence sets into one final context pool.

**Outcome**

- Complete two-stage DR-RAG retrieval pipeline.
- Improved retrieval coverage for multi-hop queries.
- Final context assembly module ready.

### Week 17: Answer Generation and API Integration

**Tasks**

- Generate final answers using the combined anchor and mined evidence.
- Add source references or chunk citations.
- Build FastAPI endpoints for document ingestion, querying, retrieval inspection, and answer generation.
- Test the full end-to-end pipeline.

**Outcome**

- End-to-end DR-RAG system completed.
- Backend API available for demo or frontend integration.
- Final answers grounded in retrieved evidence.

### Week 18: Evaluation and Baseline Comparison

**Tasks**

- Evaluate baseline RAG and DR-RAG using the same datasets and queries.
- Measure Recall@K, MRR, Exact Match, latency, Context Retention Score, and hallucination rate.
- Use RAGAS for faithfulness, answer relevancy, context precision, and context recall.
- Track experiments using MLflow.

**Outcome**

- Quantitative comparison between baseline RAG and proposed DR-RAG.
- Evaluation tables and graphs.
- Evidence of performance improvement or identified limitations.

### Week 19: Ablation Study

**Tasks**

- Test system variants by removing key components:
  - No rolling summaries.
  - Fixed-size chunking instead of CAAC.
  - No query routing.
  - No evidence mining.
- Compare performance of each configuration.
- Analyze which component contributes most to performance.

**Outcome**

- Ablation study completed.
- Component-level contribution analysis.
- Stronger research justification for the proposed architecture.

### Week 20: Final Report, Documentation, and Presentation

**Tasks**

- Finalize the project report.
- Add methodology, architecture diagrams, implementation details, results, and conclusions.
- Prepare final presentation slides.
- Prepare demo flow and explanation of system modules.
- Document limitations and future scope.

**Outcome**

- Final major project report.
- Presentation-ready system demo.
- Completed documentation and evaluation results.

## Evaluation Plan

| Evaluation Area | Metric / Method | Purpose |
|---|---|---|
| **Retrieval Quality** | Recall@K | Measures whether relevant chunks are retrieved. |
| **Ranking Quality** | Mean Reciprocal Rank | Measures how early relevant chunks appear in the result list. |
| **Answer Correctness** | Exact Match / F1 | Measures correctness against benchmark answers. |
| **RAG Quality** | RAGAS Faithfulness | Measures whether the answer is grounded in retrieved context. |
| **Answer Relevance** | RAGAS Answer Relevancy | Measures whether the generated answer addresses the query. |
| **Context Quality** | RAGAS Context Precision / Recall | Measures usefulness and completeness of retrieved context. |
| **Chunking Quality** | Context Retention Score | Measures whether chunks remain meaningful in isolation. |
| **System Efficiency** | Latency | Measures response time for baseline and DR-RAG pipelines. |
| **Reliability** | Hallucination Rate | Measures unsupported or fabricated claims in generated answers. |

## Ablation Study Plan

| Configuration | Removed Component | Purpose |
|---|---|---|
| **A1** | Rolling summaries removed | Tests whether context augmentation improves retrieval and generation. |
| **A2** | CAAC replaced with fixed-size chunking | Tests whether adaptive semantic chunking improves performance. |
| **A3** | Query router removed | Tests whether classifier-based routing reduces unnecessary two-stage retrieval. |
| **A4** | Evidence mining removed | Tests whether Stage 2 retrieval improves multi-hop question answering. |

## Expected Final Deliverables

- Working baseline RAG implementation.
- Working CAAC chunking pipeline.
- Working DR-RAG two-stage retrieval pipeline.
- FastAPI backend.
- Qdrant vector database with indexed document chunks.
- Evaluation scripts and MLflow experiment logs.
- Comparative results between baseline and proposed method.
- Ablation study results.
- Final project report.
- Final presentation and demo.

## Summary

The finalized technology stack is suitable for this project because it balances research novelty, practical implementation, local execution, and measurable evaluation. BGE-M3 and Qdrant provide strong semantic retrieval foundations. Phi-3 Mini enables efficient local semantic boundary detection and summarization. DistilBERT provides lightweight query routing. LlamaIndex simplifies retrieval orchestration, while spaCy and AllenNLP OpenIE support linguistic and evidence extraction tasks. FastAPI, MLflow, RAGAS, PyTorch, HuggingFace, and Docker complete the system by supporting deployment, training, tracking, and evaluation.
