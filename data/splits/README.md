# Benchmark Dataset Splits

This directory contains frozen, reproducible benchmark splits for evaluating STP-RAG and its ablation variants.

## 1. HotpotQA (Primary Multi-Hop Benchmark)
- **Source**: `hotpotqa/hotpot_qa` (validation distractor set).
- **Sub-directory**: `data/splits/hotpotqa/` (and root `dev.json`, `test.json`).
- **Stratification**: 50% "bridge" queries, 50% "comparison" queries.
- **Random Seed**: 42.
- **Split Sizes**:
  - `dev.json`: 150 documents, 150 QA pairs.
  - `test.json`: 150 documents, 150 QA pairs.
- **Document Structure**:
  - Each document comprises the 10 context Wikipedia paragraphs from a HotpotQA query formatted with Markdown section headers (`# <Article Title>`).
  - `gold_sentences` / `supporting_facts`: Exact sentences marked as supporting facts in the ground truth.

## 2. QASPER (Scientific Papers Benchmark)
- **Source**: AllenAI QASPER official dataset (`qasper-train-dev-v0.3.tgz`).
- **Sub-directory**: `data/splits/qasper/`.
- **Domain**: NLP and AI research papers from arXiv.
- **Random Seed**: 42.
- **Split Sizes**:
  - `qasper/dev.json`: 50 scientific papers.
  - `qasper/test.json`: 50 scientific papers.
- **Document Structure**:
  - Full papers formatted with hierarchical Markdown headers (`# Title`, `## Abstract`, `## Section`, `### Subsection`).
  - `gold_sentences`: Exact evidence paragraphs verified by NLP researchers.
