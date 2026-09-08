# STP-RAG Implementation: Phase-wise Plan & Validation Checklist

## Phase 0: Research Environment, Tooling & Scaffolding
- [x] 0.1 Create clean Python 3.11 environment in `.venv`
- [x] 0.2 Install core dependencies: `torch`, `sentence-transformers`, `spacy`, `qdrant-client`, `ragas`, `datasets`, `mlflow`, `optuna`, `scipy`, `pytest`, etc.
- [x] 0.3 Download spaCy models (`en_core_web_sm`)
- [x] 0.4 Setup project directory layout (`stp_rag/`, `notebooks/`, `configs/`, `data/{raw,splits,cache}`, `results/`, `tests/`)
- [x] 0.5 Create `docker/docker-compose.yml` for Qdrant and MLflow
- [x] 0.6 Download/prepare benchmark dataset splits (`data/splits/dev.json`, `data/splits/test.json`, `data/qasper/splits/`)
- [x] 0.7 **Phase 0 Validation:** Run environment diagnostic script verifying Python 3.11, PyTorch, spaCy, and directory structure.

---

## Phase 1: Core Modular Library (`stp_rag/`) Implementation & Unit Testing
- [x] 1.1 Ingestion & Structure Units (`stp_rag/ingestion/structure_units.py`)
  - Unit segmentation (sentences/paragraphs), cumulative token progression $p_i$, structural flags (`follows_heading`, `follows_list_item`, `is_paragraph_start`, `follows_clause_marker`).
- [x] 1.2 Embeddings & Semantic Transition Profile (`stp_rag/profile/`)
  - `embed.py`: BGE unit-normalized embedding wrapper with SQLite caching support.
  - `stp.py`: Cosine displacement $d_i$, document-relative progression $\Delta p_i$, semantic velocity $v_i$, acceleration $a_i$, volatility $\sigma_i$, and robust dev-split normalization.
  - `structural_features.py`: Scaled structural score $S_i$.
- [x] 1.3 Boundary & Granularity Chunking (`stp_rag/chunking/`)
  - `boundary_policy.py`: Logistic boundary probability $P(B_i=1)$.
  - `budget_policy.py`: Local target budget $T_i$ and clipped target length $L_i$.
  - `segment.py`: Algorithm A chunking engine enforcing $L_{\min}, L_{\max}, \tau_B$.
  - `variants.py`: Exact parameter masks for the 6 ablation rows.
- [x] 1.4 Selective Context Horizon & Propagation (`stp_rag/context/`)
  - `reference_signal.py`: Composite reference score $R_i = \max(R_i^{\text{coref}}, R_i^{\text{abbrev}}, R_i^{\text{lexcue}})$.
  - `horizon.py`: Adaptive context horizon $H_i$.
  - `selective_propagation.py`: Relevance score $\text{Rel}(c_j, c_i)$, $\tau_R$ filtering, chunk augmentation $\tilde c_i = [c_i \Vert \text{SelectedContext}_i]$.
- [x] 1.5 Evaluation Metrics, Schema & Tracking (`stp_rag/eval/`, `stp_rag/tracking/`)
  - `results_schema.py`: `ArmResult` dataclass, JSON serializer/loader.
  - `retrieval_metrics.py`: Recall@1, Recall@5, Recall@10, MRR.
  - `answer_metrics.py`: Normalized Exact Match (EM), token-level F1.
  - `efficiency_metrics.py`: Chunks/doc, tokens/chunk, build time, latency.
  - `ragas_eval.py`: RAGAS metrics wrapper.
  - `mlflow_utils.py`: MLflow experiment tracking helper.
- [x] 1.6 **Phase 1 Validation:** Comprehensive unit tests in `tests/` covering math formulas, segmentation logic, context horizon, and schema validation (`pytest tests/ -v`: 18 passing tests).

---

## Phase 2: Ingestion, Precomputation & Dev-Split Calibration
- [x] 2.1 Notebook `01_ingestion_and_embeddings.ipynb`: Ingest corpus and cache embeddings to SQLite/Parquet.
- [x] 2.2 Notebook `02_stp_computation.ipynb`: Precompute raw $v_i, a_i, \sigma_i$ and verify kinematic distributions.
- [x] 2.3 Notebook `03_calibration_dev_split.ipynb`: Fit dev normalization statistics, run Bayesian search (Optuna) for optimal $\Theta^*$ maximizing dev Recall@5, freeze to `configs/stp_params.yaml` with SHA-256 checksum.
- [x] 2.4 Notebook `04_query_router_training.ipynb`: Single-hop vs. multi-hop query router interface.
- [x] 2.5 **Phase 2 Validation:** Verify `configs/stp_params.yaml` integrity, frozen stats, and end-to-end embedding cache loading.

---

## Phase 3: Controlled 6-Arm Ablation Experiments
- [x] 3.1 `05_ablation_fixed_size.ipynb`: Fixed-size chunks baseline.
- [x] 3.2 `06_ablation_similarity_threshold.ipynb`: Semantic similarity baseline.
- [x] 3.3 `07_ablation_velocity_only.ipynb`: Velocity-only arm.
- [x] 3.4 `08_ablation_velocity_acceleration.ipynb`: Velocity + acceleration arm.
- [x] 3.5 `09_ablation_full_stp.ipynb`: Full STP (v, a, $\sigma$, S) arm.
- [x] 3.6 `10_ablation_full_stp_context.ipynb`: Full STP + selective context horizon arm.
- [x] 3.7 **Phase 3 Validation:** 36 ablation runs (18 HotpotQA + 18 QASPER across seeds 42, 123, 999) executed with frozen dev-calibrated $\Theta^*$, evaluated with Ollama local LLM generation.

---

## Phase 4: Master Compilation, Statistical Significance & Visualization
- [x] 4.1 Notebook `99_master_results_compilation.ipynb`: Compile all results across seeds.
- [x] 4.2 Compute Paired Bootstrap CIs ($B=10,000, 95\%$ CI) against velocity-only baseline.
- [x] 4.3 Generate research figures: $\Delta$ vs. baseline (`delta_vs_baseline.png`), Latency vs. Recall@5 Pareto front (`pareto_latency_recall.png`), document structure breakdown (`structure_breakdown.png`).
- [x] 4.4 Export final populated Table 2 in Markdown and LaTeX (`table2_generated.tex`, `table2_qasper.tex`).
- [x] 4.5 **Phase 4 Validation:** Verified all Table 2 and Table 3 cells are populated with real empirical data and bootstrap CIs verify statistical significance.

---

## Phase 5: Paper Integration & Final Manuscript Verification
- [x] 5.1 Update `STP_RAG_IEEE_Paper.tex` and `STP_RAG_IEEE_Paper.md` with measured results in Table 2 & Table 3.
- [x] 5.2 Synthesize discussion answering the 3 core research questions based on empirical findings.
- [x] 5.3 Overleaf-ready LaTeX package verified with BibLaTeX / Biber bibliography.
- [x] 5.4 **Phase 5 Validation:** Final end-to-end check of manuscript and reproducible artifacts.
