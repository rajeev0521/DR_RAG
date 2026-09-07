# STP-RAG: Semantic Transition Profile Guided Adaptive Chunking and Context Selection

**Semantic Transition Profile Guided Adaptive Chunking and Context Selection for Long-Document Retrieval-Augmented Generation**

---

## Overview

STP-RAG introduces the *Semantic Transition Profile* (STP) — a local representation of document dynamics that combines **semantic velocity**, **semantic acceleration**, **semantic volatility**, and **structural cues** — to guide boundary selection, target chunk length, and selective context propagation in RAG pipelines.

Unlike fixed-window or single-threshold semantic chunking, STP-RAG models the *pattern* of local semantic change to produce more useful chunk representations for long-document retrieval.

## Architecture

```
Document → Structure Units → Embeddings → STP Computation
    ↓              ↓              ↓              ↓
    └── Boundary Policy ← Velocity + Acceleration + Volatility + Structure
              ↓
    Adaptive Chunking (Algorithm A) → Chunks
              ↓
    Selective Context Propagation → Augmented Chunks
              ↓
    Vector Indexing (Qdrant) → Dense Retrieval → Answer Generation
```

## Project Structure

```
stp_rag/
├── ingestion/          # Document parsing, structure unit extraction
├── profile/            # STP computation: embeddings, velocity, acceleration, volatility
├── chunking/           # Adaptive boundary detection, segmentation engine
├── context/            # Selective context propagation and horizon
├── retrieval/          # Query routing, dense retrieval
├── indexing/           # Qdrant vector store integration
├── generation/         # LLM answer generation
├── eval/               # Evaluation metrics and results compilation
├── tracking/           # MLflow experiment tracking
└── experiments/        # End-to-end experiment pipeline

configs/                # Calibrated hyperparameters (Θ*)
notebooks/              # Reproducible research notebooks (00–99)
tests/                  # Unit and integration tests
docker/                 # Qdrant + MLflow docker-compose
scripts/                # Batch runners and notebook generators
results/                # Experimental results (JSON)
STP_RAG_Overleaf_Package/  # IEEE paper LaTeX source
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Start Infrastructure

```bash
cd docker && docker-compose up -d
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run Experiments

```bash
python scripts/run_all_experiments.py
```

## Ablation Arms

The evaluation protocol compares 6 chunking strategies across 3 random seeds:

| Arm | Description |
|-----|-------------|
| **Fixed-Size** | 256-token fixed windows (baseline) |
| **Similarity Threshold** | Adjacent cosine similarity boundary detection |
| **Velocity Only** | STP with velocity signal only |
| **Velocity + Acceleration** | STP with velocity and acceleration |
| **Full STP** | Velocity + acceleration + volatility + structure |
| **Full STP + Context** | Full STP with selective context propagation |

## Key Research Questions

1. Does modelling the *pattern* of local semantic change produce more useful RAG representations than relying on a single adjacent-unit similarity value?
2. Which STP components (velocity, acceleration, volatility, structure) contribute most to retrieval quality?
3. Does selective context propagation improve answer quality without excessive retrieval cost?

## Tech Stack

- **Embeddings:** BGE-M3 (BAAI/bge-m3)
- **Vector DB:** Qdrant
- **Experiment Tracking:** MLflow
- **Hyperparameter Search:** Optuna
- **Evaluation:** RAGAS, custom retrieval & answer metrics
- **NLP:** spaCy, sentence-transformers

## License

MIT License — see [LICENSE](LICENSE) for details.
