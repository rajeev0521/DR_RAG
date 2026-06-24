# System Architecture: Adaptive Semantic-Velocity Chunking with DR-RAG

## 1. Architecture Overview

The DR-RAG system is organized into two primary phases operating over a shared vector store, orchestrated through a FastAPI backend and tracked via MLflow.

```mermaid
graph TB
    subgraph INPUT["📄 Document Input"]
        PDF["PDF Documents"]
        TXT["Text Files"]
        MD["Markdown Files"]
    end

    subgraph PHASE1["Phase I: Context-Augmented Agentic Chunking (CAAC)"]
        INGEST["Document Ingestion<br/><i>PDF/Text Extraction</i>"]
        PREPROC["Preprocessing<br/><i>spaCy NLP Pipeline</i>"]
        SV["Semantic Velocity<br/><i>BGE-M3 Embeddings</i>"]
        BOUNDARY["Boundary Detection<br/><i>Phi-3 Mini SLM</i>"]
        REFINE["Adaptive Refinement<br/><i>Merge / Retain / Split</i>"]
        SUMMARY["Rolling Summaries<br/><i>Phi-3 Mini SLM</i>"]
        BUDGET["Token Budget Scaling<br/><i>Velocity-Driven</i>"]
        EMBED["Chunk Embedding<br/><i>BGE-M3</i>"]
    end

    subgraph VECTORDB["🗄️ Vector Store"]
        QDRANT["Qdrant<br/><i>HNSW Index + Metadata</i>"]
    end

    subgraph PHASE2["Phase II: Dynamic-Relevant RAG (DR-RAG)"]
        ROUTER["Query Router<br/><i>DistilBERT Classifier</i>"]
        ANCHOR["Stage 1: Anchor Retrieval<br/><i>Cosine Similarity Top-K</i>"]
        EXTRACT["Evidence Extraction<br/><i>AllenNLP OpenIE</i>"]
        MINING_Q["Mining Query Generation<br/><i>Query + Facts Synthesis</i>"]
        MINING_R["Stage 2: Evidence Mining<br/><i>Expanded Retrieval</i>"]
        ASSEMBLE["Context Assembly<br/><i>Dedup + Rank</i>"]
        GENERATE["Answer Generation<br/><i>LLM Response</i>"]
    end

    subgraph INFRA["⚙️ Infrastructure"]
        API["FastAPI Backend"]
        MLFLOW["MLflow Tracking"]
        DOCKER["Docker Containers"]
        RAGAS["RAGAS Evaluation"]
    end

    subgraph OUTPUT["📤 Response"]
        ANSWER["Structured Answer<br/><i>Answer + Citations + Confidence</i>"]
    end

    INPUT --> INGEST
    INGEST --> PREPROC
    PREPROC --> SV
    SV --> BOUNDARY
    BOUNDARY --> REFINE
    REFINE --> SUMMARY
    SUMMARY --> BUDGET
    BUDGET --> EMBED
    EMBED --> QDRANT

    QDRANT --> ANCHOR
    ROUTER --> ANCHOR
    ANCHOR -->|"Single-Hop"| GENERATE
    ANCHOR -->|"Multi-Hop"| EXTRACT
    EXTRACT --> MINING_Q
    MINING_Q --> MINING_R
    MINING_R --> ASSEMBLE
    ASSEMBLE --> GENERATE
    GENERATE --> ANSWER

    API -.->|"Orchestrates"| PHASE1
    API -.->|"Orchestrates"| PHASE2
    MLFLOW -.->|"Tracks"| PHASE1
    MLFLOW -.->|"Tracks"| PHASE2
    RAGAS -.->|"Evaluates"| GENERATE
    DOCKER -.->|"Hosts"| QDRANT
    DOCKER -.->|"Hosts"| API
```

---

## 2. Component Inventory

### 2.1 Phase I Components (Indexing Pipeline)

| Component | Module | Responsibility | Key Inputs | Key Outputs |
|---|---|---|---|---|
| **Document Ingestion** | `src/ingestion/` | Extract raw text from PDF, TXT, and MD files | Raw document files | Extracted text with page metadata |
| **Preprocessing** | `src/ingestion/` | Sentence splitting, tokenization, metadata extraction | Extracted text | Sentence list with metadata (page, section, position) |
| **Semantic Velocity** | `src/chunking/` | Compute embedding drift between adjacent sentences | Sentence embeddings (BGE-M3) | Velocity scores per sentence boundary |
| **Boundary Detection** | `src/chunking/` | Identify optimal semantic chunk boundaries | Velocity scores + document structure + SLM analysis | Boundary positions |
| **Adaptive Refinement** | `src/chunking/` | Merge low-velocity, retain moderate, split high-velocity regions | Initial chunks + velocity scores | Refined semantic chunks |
| **Rolling Summaries** | `src/chunking/` | Generate context summaries from preceding chunks | Previous 3 chunks | Summary text |
| **Token Budget Scaling** | `src/chunking/` | Scale summary length by semantic velocity magnitude | Velocity scores + summaries | Budget-scaled summaries |
| **Chunk Embedding** | `src/chunking/` | Embed augmented chunks (summary ⊕ chunk) into dense vectors | Augmented chunk text | Dense vector embeddings |

### 2.2 Phase II Components (Retrieval Pipeline)

| Component | Module | Responsibility | Key Inputs | Key Outputs |
|---|---|---|---|---|
| **Query Router** | `src/retrieval/` | Classify queries as single-hop or multi-hop | User query | Query class + routing decision |
| **Anchor Retrieval** | `src/retrieval/` | Retrieve top-K most similar chunks | Query embedding + Qdrant index | Anchor chunk set with scores |
| **Evidence Extraction** | `src/retrieval/` | Extract structured triples/propositions from anchors | Anchor chunks | Subject-relation-object triples |
| **Mining Query Gen** | `src/retrieval/` | Synthesize expanded query from original + extracted facts | Original query + intermediate facts | Mining query string |
| **Evidence Mining** | `src/retrieval/` | Second retrieval pass using mining query | Mining query embedding + Qdrant index | Evidence chunk set (deduplicated) |
| **Context Assembly** | `src/retrieval/` | Merge, deduplicate, and rank anchor + evidence chunks | Anchor set + evidence set | Final context pool |
| **Answer Generation** | `src/retrieval/` | Generate grounded answer from context pool | Query + context pool | Structured response |

### 2.3 Infrastructure Components

| Component | Module | Responsibility |
|---|---|---|
| **FastAPI Backend** | `src/api/` | REST API endpoints for ingestion, querying, and evaluation |
| **Qdrant Vector DB** | Docker service | HNSW-indexed vector storage with metadata filtering |
| **MLflow Tracking** | Infrastructure | Experiment tracking, metric logging, model versioning |
| **RAGAS Evaluation** | `src/evaluation/` | RAG-specific quality metrics (faithfulness, relevancy, precision, recall) |

---

## 3. Technology Mapping

```mermaid
graph LR
    subgraph MODELS["🧠 Models"]
        BGEM3["BGE-M3<br/><i>Embedding</i>"]
        PHI3["Phi-3 Mini<br/><i>Boundary + Summary</i>"]
        DISTIL["DistilBERT<br/><i>Query Classification</i>"]
    end

    subgraph FRAMEWORKS["📦 Frameworks"]
        LLAMA["LlamaIndex<br/><i>RAG Orchestration</i>"]
        SPACY["spaCy<br/><i>NLP Processing</i>"]
        ALLEN["AllenNLP OpenIE<br/><i>Triple Extraction</i>"]
        PT["PyTorch + HuggingFace<br/><i>Model Training</i>"]
    end

    subgraph INFRA2["🏗️ Infrastructure"]
        FAST["FastAPI<br/><i>Backend API</i>"]
        QD["Qdrant<br/><i>Vector Database</i>"]
        ML["MLflow<br/><i>Experiment Tracking</i>"]
        DK["Docker<br/><i>Containerization</i>"]
        RG["RAGAS<br/><i>Evaluation</i>"]
    end

    BGEM3 -->|"Embeds sentences & chunks"| LLAMA
    PHI3 -->|"Detects boundaries & generates summaries"| LLAMA
    DISTIL -->|"Routes queries"| LLAMA
    SPACY -->|"Preprocesses text"| LLAMA
    ALLEN -->|"Extracts evidence triples"| LLAMA
    PT -->|"Trains/fine-tunes models"| DISTIL
    LLAMA -->|"Stores/retrieves vectors"| QD
    FAST -->|"Serves"| LLAMA
    ML -->|"Tracks experiments"| FAST
    DK -->|"Hosts"| QD
    DK -->|"Hosts"| FAST
    RG -->|"Evaluates outputs"| ML
```

---

## 4. Data Flow

### 4.1 Indexing Data Flow (Phase I)

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Ingest as Ingestion Module
    participant NLP as spaCy Pipeline
    participant Embed as BGE-M3 Embedder
    participant SLM as Phi-3 Mini
    participant Chunk as Chunking Engine
    participant VDB as Qdrant

    User->>API: Upload document (PDF/TXT/MD)
    API->>Ingest: Extract raw text
    Ingest->>NLP: Sentence splitting + tokenization
    NLP-->>Ingest: Sentences with metadata

    Ingest->>Embed: Embed individual sentences
    Embed-->>Ingest: Sentence embeddings

    Ingest->>Chunk: Compute semantic velocity
    Note over Chunk: cosine_distance(emb[i], emb[i+1])

    Chunk->>SLM: Analyze document for boundary suggestions
    SLM-->>Chunk: Suggested boundaries

    Note over Chunk: Combine velocity signals +<br/>SLM boundaries + structural cues

    Chunk->>Chunk: Adaptive refinement<br/>(merge low / retain mid / split high)

    loop For each chunk C_i
        Chunk->>SLM: Generate rolling summary of C_{i-3}...C_{i-1}
        SLM-->>Chunk: Summary text
        Chunk->>Chunk: Scale summary by velocity budget
        Chunk->>Chunk: Augment: C'_i = summary ⊕ C_i
    end

    Chunk->>Embed: Embed augmented chunks
    Embed-->>Chunk: Chunk embeddings

    Chunk->>VDB: Store embeddings + metadata
    Note over VDB: doc_id, chunk_id, page,<br/>section, velocity_score, doc_type

    VDB-->>API: Indexing complete
    API-->>User: Success response
```

### 4.2 Retrieval Data Flow (Phase II)

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Router as DistilBERT Router
    participant Embed as BGE-M3 Embedder
    participant VDB as Qdrant
    participant OIE as AllenNLP OpenIE
    participant SLM as Phi-3 Mini
    participant Gen as Generation LLM

    User->>API: Submit query Q
    API->>Router: Classify query complexity
    Router-->>API: "single_hop" | "multi_hop"

    API->>Embed: Embed query Q
    Embed-->>API: Query vector q

    API->>VDB: Stage 1 — Retrieve top-K (q)
    VDB-->>API: Anchor chunks A = {A_1...A_k}

    alt Single-Hop Query
        API->>Gen: Generate(Q, A)
        Gen-->>API: Answer R
    else Multi-Hop Query
        API->>OIE: Extract triples from A
        OIE-->>API: Intermediate facts F

        opt Noisy extraction
            API->>SLM: Clean/refine facts F
            SLM-->>API: Refined facts F'
        end

        API->>API: Synthesize mining query Q_mine = Q ⊕ F'
        API->>Embed: Embed Q_mine
        Embed-->>API: Mining vector q_mine

        API->>VDB: Stage 2 — Retrieve top-K (q_mine) \ A
        VDB-->>API: Evidence chunks E = {E_1...E_m}

        API->>API: Assemble context: Ctx = A ∪ E
        API->>Gen: Generate(Q, Ctx)
        Gen-->>API: Answer R
    end

    API-->>User: Structured response<br/>(answer + citations + confidence)
```

---

## 5. Module Directory Structure

The source code is organized to mirror the two-phase architecture:

```
src/
├── ingestion/                  # Phase I: Document ingestion
│   ├── __init__.py
│   ├── extractors.py           # PDF, TXT, MD text extraction
│   ├── preprocessor.py         # spaCy sentence splitting, metadata
│   └── document.py             # Document data model
│
├── chunking/                   # Phase I: CAAC pipeline
│   ├── __init__.py
│   ├── semantic_velocity.py    # Embedding drift computation
│   ├── boundary_detector.py    # SLM-guided boundary detection
│   ├── adaptive_refiner.py     # Merge/retain/split logic
│   ├── rolling_summary.py      # Summary generation + budget scaling
│   ├── chunk_embedder.py       # Augmented chunk embedding
│   └── models.py               # Chunk data models
│
├── retrieval/                  # Phase II: DR-RAG pipeline
│   ├── __init__.py
│   ├── query_router.py         # DistilBERT query classifier
│   ├── anchor_retrieval.py     # Stage 1 retrieval
│   ├── evidence_extractor.py   # AllenNLP OpenIE extraction
│   ├── mining_query.py         # Mining query synthesis
│   ├── evidence_miner.py       # Stage 2 retrieval
│   ├── context_assembler.py    # Dedup + rank + assemble
│   └── generator.py            # Answer generation
│
├── evaluation/                 # Evaluation framework
│   ├── __init__.py
│   ├── ragas_evaluator.py      # RAGAS metrics wrapper
│   ├── retrieval_metrics.py    # Recall@K, MRR computation
│   └── experiment_tracker.py   # MLflow integration
│
└── api/                        # FastAPI backend
    ├── __init__.py
    ├── main.py                 # App entry point
    ├── routes/
    │   ├── ingest.py           # Document upload endpoints
    │   ├── query.py            # Query endpoints
    │   └── evaluate.py         # Evaluation endpoints
    └── schemas.py              # Pydantic request/response models
```

---

## 6. Inter-Component Communication

All components communicate through well-defined Python data classes and Pydantic models. The system uses **synchronous in-process calls** for the prototype, with the option to introduce message queues for production scaling.

| Interface | From | To | Data Format |
|---|---|---|---|
| Document Upload | API | Ingestion | `DocumentUpload` Pydantic model |
| Preprocessed Text | Ingestion | Chunking | `PreprocessedDocument` dataclass (sentences + metadata) |
| Chunk Store | Chunking | Qdrant | `ChunkPayload` (embedding vector + metadata dict) |
| Query Request | API | Retrieval | `QueryRequest` Pydantic model |
| Retrieval Result | Qdrant | Retrieval | `ScoredPoint` (Qdrant native type) |
| Structured Response | Retrieval | API | `QueryResponse` Pydantic model |
| Evaluation Request | API | Evaluation | `EvaluationRequest` Pydantic model |
| Experiment Log | Evaluation | MLflow | MLflow tracking API |

---

## 7. Deployment Architecture

```mermaid
graph TB
    subgraph DOCKER["Docker Compose Environment"]
        subgraph APP["Application Container"]
            FASTAPI["FastAPI Server<br/><i>Port 8000</i>"]
            MODELS_CACHE["Model Cache<br/><i>BGE-M3, Phi-3, DistilBERT</i>"]
        end

        subgraph VDB_CONTAINER["Qdrant Container"]
            QDRANT_SVC["Qdrant Server<br/><i>REST: 6333 | gRPC: 6334</i>"]
            STORAGE["Persistent Storage<br/><i>./data/qdrant_data</i>"]
        end

        subgraph TRACKING["MLflow Container"]
            MLFLOW_SVC["MLflow Server<br/><i>Port 5000</i>"]
            ARTIFACTS["Artifact Store"]
        end
    end

    CLIENT["Client / Frontend"] -->|"HTTP"| FASTAPI
    FASTAPI -->|"REST / gRPC"| QDRANT_SVC
    FASTAPI -->|"Tracking API"| MLFLOW_SVC
    QDRANT_SVC --- STORAGE
    MLFLOW_SVC --- ARTIFACTS
```
