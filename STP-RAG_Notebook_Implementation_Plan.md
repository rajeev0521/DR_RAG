# STP-RAG: Notebook-Style Implementation Plan

This restructures the earlier plan into a **utils package + experiment notebooks + master compilation notebook** layout. The core rule that keeps this maintainable: **notebooks never contain pipeline logic, only orchestration.** Every formula, model call, and metric lives in `stp_rag/`; a notebook imports it, runs it on a config, logs to MLflow, and writes one results file. That's what lets the master notebook compile results mechanically instead of re-deriving them.

---

## 1. Repository Structure

```
stp-rag/
├── stp_rag/                          # ALL logic lives here — importable, unit-testable, shared by every notebook
│   ├── ingestion/
│   │   ├── structure_units.py        # ExtractStructureAwareUnits (spaCy)
│   │   └── parsers.py
│   ├── profile/
│   │   ├── embed.py                  # BGE-M3 wrapper
│   │   ├── stp.py                    # v_i, a_i, sigma_i, normalization fit/apply
│   │   └── structural_features.py    # S_i
│   ├── chunking/
│   │   ├── boundary_policy.py        # P(B_i=1)
│   │   ├── budget_policy.py          # T_i, L_i
│   │   ├── segment.py                # Segment(U, P, Θ, L_min, L_max)
│   │   └── variants.py               # one function per Table 1 row, same code path
│   ├── context/
│   │   ├── horizon.py                # H_i
│   │   ├── reference_signal.py       # R_i
│   │   └── selective_propagation.py  # c̃_i = c_i ‖ SelectedContext_i
│   ├── boundary_slm/
│   │   └── phi3_assist.py            # Phi-3 Mini rolling summaries / boundary hints
│   ├── indexing/
│   │   └── qdrant_store.py           # one Qdrant collection per ablation arm
│   ├── retrieval/
│   │   ├── query_router.py           # DistilBERT single/multi-hop classifier
│   │   ├── stage1_dense.py           # LlamaIndex over Qdrant
│   │   ├── stage2_openie.py          # AllenNLP OpenIE triples
│   │   └── fusion.py
│   ├── generation/
│   │   └── answer.py                 # prompt assembly, logs exact prompt text
│   ├── eval/
│   │   ├── retrieval_metrics.py      # Recall@1/5/10, MRR
│   │   ├── answer_metrics.py         # EM, token-F1
│   │   ├── efficiency_metrics.py     # chunks/doc, tokens/chunk, build time, latency
│   │   ├── ragas_eval.py
│   │   └── results_schema.py         # ONE dataclass every notebook writes results through
│   └── tracking/
│       └── mlflow_utils.py           # start_run(), log_config(), log_result() wrappers
│
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_ingestion_and_embeddings.ipynb        # builds & caches embeddings ONCE, shared by all arms
│   ├── 02_stp_computation.ipynb                 # computes v, a, sigma over cached embeddings
│   ├── 03_calibration_dev_split.ipynb           # fits Θ on dev split, freezes configs/stp_params.yaml
│   ├── 04_query_router_training.ipynb           # fine-tunes DistilBERT router
│   ├── 05_ablation_fixed_size.ipynb             ┐
│   ├── 06_ablation_similarity_threshold.ipynb   │  each: build chunks for
│   ├── 07_ablation_velocity_only.ipynb          │  this arm → index → retrieve
│   ├── 08_ablation_velocity_acceleration.ipynb  │  → generate → eval → write
│   ├── 09_ablation_full_stp.ipynb               │  one results file
│   ├── 10_ablation_full_stp_context.ipynb       ┘
│   └── 99_master_results_compilation.ipynb      # reads every results file → Table 1 & Table 2 → plots
│
├── configs/
│   ├── stp_params.yaml               # frozen after 03_calibration_dev_split.ipynb
│   ├── retrieval.yaml                # shared index/retriever settings
│   └── eval.yaml                     # seeds, split definitions, metric config
├── data/
│   ├── raw/
│   ├── splits/                       # dev.json / test.json — frozen once, never regenerated ad hoc
│   └── cache/                        # cached embeddings, STP arrays (pickled/parquet, keyed by doc_id)
├── results/
│   └── <system_name>__seed<k>.json   # one file per (arm, seed) — see schema in §3
├── mlruns/                           # MLflow tracking store
├── docker/
│   ├── docker-compose.yml            # qdrant + mlflow + jupyter
│   └── Dockerfile
└── tests/                            # pytest against stp_rag/, independent of notebooks
```

---

## 2. Why this split (and the one rule that makes it work)

- **`stp_rag/` is the only place formulas exist.** If $v_i$'s formula changes, you edit `profile/stp.py` once; every ablation notebook picks it up automatically because they all `import stp_rag.profile.stp`. No notebook should ever redefine a function inline — that's the failure mode that makes ablations silently diverge from each other.
- **Notebooks 05–10 are near-identical.** Each one is: load frozen config → call `stp_rag.chunking.variants.<arm>()` → index → retrieve → generate → evaluate → `results_schema.write(...)`. You can literally copy notebook 05 to make notebook 06 and change one import + one config key. That's intentional — it's what guarantees the six arms are comparable.
- **The master notebook does zero computation of its own.** It only reads `results/*.json` (or queries MLflow by experiment tag) and formats. If you ever catch yourself computing a metric inside `99_master_results_compilation.ipynb`, that logic belongs in `stp_rag/eval/` instead.

---

## 3. Shared Results Schema (`stp_rag/eval/results_schema.py`)

Every ablation notebook must write results through this schema — it's what makes the master notebook a pure aggregator instead of six different bespoke parsers:

```python
@dataclass
class ArmResult:
    system: str            # "fixed_size" | "similarity_threshold" | "velocity_only" |
                            # "velocity_acceleration" | "full_stp" | "full_stp_context"
    seed: int
    split: str              # "dev" | "test"
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    em: float
    f1: float
    chunks_per_doc: float
    tokens_per_chunk: float
    build_time_sec: float
    retrieval_latency_ms: float
    ragas_faithfulness: float
    ragas_answer_relevancy: float
    ragas_context_precision: float
    ragas_context_recall: float
    mlflow_run_id: str       # provenance — master notebook can always trace back to the run
    embedding_model_version: str
    corpus_version: str

def write(result: ArmResult, results_dir="results/") -> Path: ...
def load_all(results_dir="results/") -> pd.DataFrame: ...
```

`load_all()` is the only function `99_master_results_compilation.ipynb` calls to get everything into one DataFrame.

---

## 4. Notebook-by-Notebook Plan

### `00_environment_setup.ipynb`
Spin up Docker services (Qdrant, MLflow) from the notebook via `subprocess`/`docker-compose up -d`; verify connectivity; pull BGE-M3 / Phi-3 Mini / DistilBERT checkpoints. No experiment logic.

### `01_ingestion_and_embeddings.ipynb`
Run `stp_rag.ingestion.structure_units` + `stp_rag.profile.embed` over the full corpus **once**. Cache to `data/cache/embeddings.parquet`. Every later notebook loads this cache instead of re-embedding — this is what makes the six ablation arms genuinely share the same embedding model, not just the same *config* for it.

### `02_stp_computation.ipynb`
Load cached embeddings → compute $d_i, v_i, a_i, \sigma_i$ over the full corpus (dev + test) → cache raw (unnormalized) STP arrays. Sanity-check plots: velocity/acceleration distributions, a manual spot-check that high-$\sigma$ regions correspond to visibly volatile passages.

### `03_calibration_dev_split.ipynb`
**Dev-split only.** Fit normalization stats for $\hat v, \hat a, \hat\sigma$; grid/Bayesian-search $w_v,w_a,w_\sigma,w_s,b,\tau_B,L_{\min},L_{\max},\alpha,\beta,\gamma,H_{\min},H_{\max},\lambda_1,\lambda_2,\lambda_3$ and the context-attachment threshold. Last cell writes `configs/stp_params.yaml` and prints its checksum. **This checksum gets pasted into every ablation notebook's first cell as an assertion** — that's the enforced dev/test wall.

### `04_query_router_training.ipynb`
Fine-tune DistilBERT (single-hop vs multi-hop) via `stp_rag.training` (or a `training/` subpackage under `stp_rag/`), log the training run to MLflow, save the checkpoint path into `configs/retrieval.yaml`.

### `05`–`10`: The Six Ablation Notebooks
Each follows the same four cells:
1. **Load & assert** — load `configs/stp_params.yaml`, assert its checksum matches the frozen one from `03_calibration_dev_split.ipynb`.
2. **Build** — call the one `stp_rag.chunking.variants` function for this arm (differs only in which STP components are zeroed out) → `selective_propagation` (only for arms 09/10) → index into a **dedicated Qdrant collection** named after the arm.
3. **Run** — for each of ≥3 seeds: route queries (`query_router`) → retrieve (`stage1_dense`, `stage2_openie`+`fusion` for multi-hop) → generate (`generation.answer`, logging the exact prompt) → score (`eval.*`).
4. **Log** — `mlflow_utils.log_result()` + `results_schema.write(ArmResult(...))` per seed.

| Notebook | Arm | STP components active |
|---|---|---|
| `05_ablation_fixed_size` | Fixed-size chunks | none |
| `06_ablation_similarity_threshold` | Similarity-threshold | distance only |
| `07_ablation_velocity_only` | Velocity-only | $v$ |
| `08_ablation_velocity_acceleration` | Velocity + acceleration | $v, a$ |
| `09_ablation_full_stp` | Full STP | $v, a, \sigma$ |
| `10_ablation_full_stp_context` | Full STP + selective context | $v, a, \sigma$ + context horizon |

### `99_master_results_compilation.ipynb`
1. `df = results_schema.load_all("results/")`
2. Group by `system`, aggregate mean ± std across seeds for every metric → this **is** Table 2, formatted directly from `df`, not retyped.
3. Cross-check against MLflow via `mlflow_run_id` for full provenance (params, exact prompts, artifacts) on any row you want to drill into.
4. Generate the three discussion plots the paper's protocol calls for: (a) metric-by-metric delta vs. velocity-only baseline, (b) quality-vs-latency/index-size scatter across arms, (c) a breakdown by document-structure type (stable narrative vs. definition-dense), if that tag is available per document.
5. Export final Table 1/Table 2 as markdown/LaTeX directly from the DataFrame for drop-in to the paper.

---

## 5. Suggested Timeline (unchanged phases, mapped to notebooks)

| Week | Notebooks |
|---|---|
| 1 | `00`, `01` |
| 2 | `02`, `03` |
| 3 | `04`; start scaffolding `stp_rag/chunking`, `stp_rag/context` used by `05`–`10` |
| 4 | `05`, `06`, `07` (simpler arms first — validates the shared pipeline before adding STP complexity) |
| 5 | `08`, `09`, `10` |
| 6 | Full re-run, ≥3 seeds each, on test split; `99` compiled iteratively as arms finish |

---

## 6. Guardrails Worth Keeping

- **Never hand-edit `results/*.json`.** If a number's wrong, fix the notebook/util and rerun — the master notebook should never need to trust anything but generated files.
- **Pin the embedding cache.** If `01_ingestion_and_embeddings.ipynb` is ever rerun (e.g., BGE-M3 version bump), bump `embedding_model_version` and keep the old cache — old results stay reproducible and comparable by version, not silently overwritten.
- **One Qdrant collection per arm**, never a shared collection with a "variant" filter — keeps index-build-time and index-size metrics per arm honest and avoids cross-arm contamination during debugging.
- Keep `tests/` running against `stp_rag/` independent of the notebooks (e.g., a synthetic 20-unit document with a hand-checked boundary) so a broken formula fails fast in CI rather than surfacing as a weird Table 2 number three weeks in.

---

**Next step:** I can scaffold `stp_rag/profile/stp.py` + `stp_rag/eval/results_schema.py` + a `05_ablation_fixed_size.ipynb` skeleton so you have one working end-to-end slice to clone for the other five.
