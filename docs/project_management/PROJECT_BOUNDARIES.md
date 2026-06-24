# Project Boundaries: DR-RAG Framework

## 1. In-Scope

### 1.1 Document Processing

| Aspect | Scope |
|---|---|
| **Input Formats** | Text-based PDF, plain text (.txt), Markdown (.md) |
| **Document Types** | Legal contracts, academic papers, technical reports, policy documents |
| **Language** | English only |
| **Document Length** | Single documents up to ~100 pages / ~50,000 tokens |
| **Batch Processing** | Sequential document ingestion (one document at a time) |

### 1.2 Chunking (Phase I — CAAC)

| Aspect | Scope |
|---|---|
| **Chunking Strategy** | Adaptive semantic chunking with SLM-guided boundary detection |
| **Semantic Velocity** | Cosine distance between adjacent sentence window embeddings |
| **Rolling Summaries** | Context augmentation from previous 3 chunks via Phi-3 Mini |
| **Token Budget Scaling** | Velocity-driven summary length adjustment |
| **Embedding Model** | BGE-M3 (1024-dimensional dense vectors) |
| **Chunk Size Range** | 64–512 tokens per chunk |
| **Metadata** | Document ID, chunk ID, page, section, velocity score, document type |

### 1.3 Retrieval (Phase II — DR-RAG)

| Aspect | Scope |
|---|---|
| **Query Routing** | Binary classification (single-hop / multi-hop) via DistilBERT |
| **Stage 1 Retrieval** | Top-K cosine similarity search over Qdrant |
| **Evidence Extraction** | AllenNLP OpenIE triple extraction from anchor chunks |
| **Stage 2 Retrieval** | Expanded mining query retrieval with deduplication |
| **Context Assembly** | Composite scoring, deduplication, positional ordering |
| **Answer Generation** | LLM-generated answer with source citations |

### 1.4 Evaluation

| Aspect | Scope |
|---|---|
| **Retrieval Metrics** | Recall@K, Mean Reciprocal Rank (MRR) |
| **Answer Quality** | Exact Match, F1 Score |
| **RAG Metrics** | RAGAS Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| **Chunking Metrics** | Context Retention Score (CRS) |
| **System Metrics** | End-to-end latency, per-stage latency |
| **Reliability** | Hallucination rate measurement |
| **Ablation Study** | 4 configurations (A1–A4) removing individual components |
| **Baseline Comparison** | Fixed-chunk RAG vs. proposed DR-RAG |

### 1.5 Infrastructure

| Aspect | Scope |
|---|---|
| **Backend API** | FastAPI REST endpoints for ingestion, querying, and evaluation |
| **Vector Database** | Qdrant (Docker container, local deployment) |
| **Experiment Tracking** | MLflow for metrics, parameters, and run comparison |
| **Containerization** | Docker Compose for Qdrant + FastAPI |

---

## 2. Out of Scope

The following capabilities are explicitly **not** part of this project:

### 2.1 Document Processing Exclusions

| Exclusion | Rationale |
|---|---|
| **Scanned/image-based PDFs** | OCR integration adds complexity without advancing the core research contributions (chunking + retrieval). |
| **Non-English documents** | BGE-M3 supports multilingual text, but evaluation datasets (HotpotQA, CUAD) are English-only. Multilingual evaluation would require separate benchmarks. |
| **Multi-modal input** (images, tables, charts) | The focus is on text-based semantic chunking. Table/image extraction is a separate research problem. |
| **Real-time streaming documents** | The system processes static, complete documents. Live document feeds are out of scope. |
| **Document generation or editing** | The system retrieves and generates answers, not documents. |

### 2.2 Retrieval Exclusions

| Exclusion | Rationale |
|---|---|
| **Graph-based retrieval** (GraphRAG, knowledge graphs) | The project's contribution is demonstrating that two-stage retrieval with evidence mining achieves comparable multi-hop performance at lower cost. |
| **Iterative/agentic retrieval loops** | DR-RAG uses exactly two retrieval stages. Unbounded iterative retrieval is not implemented. |
| **Cross-document retrieval** | Queries are answered from a single ingested document corpus. Cross-corpus federation is out of scope. |
| **Hybrid search** (sparse + dense) | The system uses dense retrieval only (BGE-M3 + cosine similarity). BM25 or sparse retrieval fusion is not included. |

### 2.3 System Exclusions

| Exclusion | Rationale |
|---|---|
| **Production deployment** | The system is a research prototype. Production concerns (horizontal scaling, authentication, rate limiting) are not addressed. |
| **Frontend / UI** | No web interface is built. Interaction is via API endpoints and evaluation scripts. |
| **Fine-tuning the generation LLM** | The generation model is used via API or inference. Fine-tuning it on domain data is not in scope. |
| **Distributed vector search** | Qdrant runs as a single-node Docker container. Cluster mode is not configured. |
| **Cost optimization** | API costs for SLM/LLM inference are not optimized (e.g., no prompt caching, no batch inference). |

---

## 3. Assumptions

### 3.1 Data Assumptions

| Assumption | Impact if Violated |
|---|---|
| Input documents contain extractable text (not images or scans). | Chunking pipeline will produce empty or garbage chunks. |
| Documents have some structural markers (headings, paragraphs, numbered sections). | SLM boundary detection may underperform; system falls back to velocity-only boundaries. |
| English text with standard grammar and vocabulary. | spaCy sentence splitting and OpenIE extraction quality will degrade for non-standard text. |
| Documents are ≤ 100 pages. | Memory constraints may arise during full-document SLM processing with Phi-3 Mini. |

### 3.2 Infrastructure Assumptions

| Assumption | Impact if Violated |
|---|---|
| Docker is available on the development machine. | Qdrant must be installed natively or hosted externally. |
| GPU is available for model inference (Phi-3 Mini, BGE-M3, DistilBERT). | CPU inference is possible but significantly slower; latency targets may not be met. |
| Sufficient disk space for model weights (~10 GB for all models). | Models must be quantized or hosted remotely. |
| Stable internet for initial model downloads from HuggingFace Hub. | Pre-download models and cache locally before starting development. |

### 3.3 Evaluation Assumptions

| Assumption | Impact if Violated |
|---|---|
| HotpotQA and CUAD datasets are accessible and usable. | Alternative evaluation datasets must be sourced. |
| RAGAS framework is compatible with the project's LLM and retrieval setup. | Custom evaluation scripts must replace RAGAS. |
| Baseline RAG produces measurable, non-trivial results. | Baseline must be debugged before meaningful comparison is possible. |

---

## 4. Dependencies

### 4.1 External Model Dependencies

| Dependency | Source | Version Strategy | Risk |
|---|---|---|---|
| **BGE-M3** | HuggingFace Hub (`BAAI/bge-m3`) | Pin to specific revision | Low — well-maintained model |
| **Phi-3 Mini** | HuggingFace Hub (`microsoft/Phi-3-mini-4k-instruct`) | Pin to specific revision | Low — Microsoft maintained |
| **DistilBERT** | HuggingFace Hub (`distilbert-base-uncased`) | Pin to specific revision | Low — very stable base model |

### 4.2 Library Dependencies

| Library | Purpose | Risk |
|---|---|---|
| `llama-index` | RAG orchestration | Medium — active development, breaking changes possible |
| `qdrant-client` | Vector DB client | Low — stable API |
| `transformers` | Model loading and inference | Low — well-maintained |
| `torch` | ML framework | Low — stable |
| `spacy` | NLP processing | Low — stable |
| `allennlp` | OpenIE extraction | Medium — less actively maintained; may need compatibility fixes |
| `ragas` | Evaluation metrics | Medium — newer library, API may change |
| `fastapi` | Backend API | Low — stable |
| `mlflow` | Experiment tracking | Low — stable |

### 4.3 Infrastructure Dependencies

| Dependency | Deployment | Risk |
|---|---|---|
| **Qdrant** | Docker container | Low — official Docker image |
| **Docker** | Host machine | Low — widely available |
| **CUDA** (optional) | GPU drivers | Medium — version compatibility with PyTorch |

---

## 5. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Phi-3 Mini produces low-quality boundaries for certain document types | Medium | High | Use structural signals as primary; SLM as refinement. Multiple prompt templates per domain. |
| AllenNLP OpenIE produces noisy/incomplete triples | Medium | Medium | SLM refinement step. Fallback to spaCy NER. Skip Stage 2 if extraction fails. |
| DistilBERT query router misclassifies queries | Low | Medium | Conservative fallback to multi-hop. Low-confidence threshold routing. |
| Qdrant latency exceeds targets at scale | Low | Low | Tune HNSW parameters. Limit corpus size for prototype. |
| RAGAS framework incompatibility | Medium | Medium | Implement custom evaluation functions as backup. |
| GPU memory insufficient for concurrent model loading | Medium | Medium | Load models sequentially. Use quantized model variants. |
| Evaluation datasets not representative of target domains | Medium | High | Supplement with manually curated legal/academic test sets (~50 examples). |

---

## 6. Constraints

### 6.1 Hardware Constraints

- **Development**: Single machine with GPU (consumer-grade, e.g., RTX 3060/4060 with 8–12 GB VRAM)
- **Memory**: ≥ 16 GB RAM for concurrent model loading and Qdrant
- **Storage**: ≥ 50 GB free disk space (models + data + Qdrant storage)

### 6.2 Timeline Constraints

- 20-week development timeline (see [PROJECT_PLAN.md](file:///c:/Users/rajee/Desktop/Major%20project/docs/project_management/PROJECT_PLAN.md))
- Research prototype, not production system
- Academic project with report and presentation deadline

### 6.3 Cost Constraints

- Local-first design: BGE-M3, Phi-3 Mini, DistilBERT all run locally
- Optional cloud LLM API for answer generation (GPT-4o-mini or Claude 3.5 Haiku)
- No managed cloud infrastructure costs (all services run in Docker)
