# DR_RAG / STP-RAG — Module-Wise Codebase Review

Reviewed by cloning `https://github.com/rajeev0521/DR_RAG.git` (main) and reading every module, notebook, config, and result file directly — this is not a description of the plan, it's what the code and generated artifacts actually do right now.

**Headline finding:** the architecture is faithfully built — the STP math, segmentation algorithm, and 6-arm ablation restriction table all match the methodology spec closely and cleanly. But the numbers currently sitting in `results/*.json` are **not valid experimental results** — they come from a 2-document smoke-test corpus, a generator that was silently offline, and a metric (RAGAS) that was never invoked. Nothing is fabricated (every gap fails to a `0.0` or a trivial `1.0`, not a made-up number), but the repo is not yet ready to populate the paper's Table 2. That's the ASAP-prompt's job.

---

## Module-by-Module Status

| Module | Files | Status | Key Finding |
|---|---|---|---|
| `ingestion/` | `structure_units.py`, `prepare_dataset.py` | ✅ Solid | Structure-aware unit extraction with token cumulative positions and structural flags looks complete and matches §1/§2.6 of the spec. |
| `profile/` | `embed.py`, `stp.py`, `structural_features.py` | ✅ Solid | `stp.py` is a **correct, faithful** implementation of $d_i, \Delta p_i, v_i, a_i, \sigma_i$ and robust median/IQR normalization — this is the strongest module in the repo. Minor risk: `compute_stp()` silently self-fits normalization when `norm_stats=None` instead of raising — a footgun if ever called on a test split without explicitly passing frozen stats. |
| `chunking/` | `boundary_policy.py`, `budget_policy.py`, `segment.py`, `variants.py` | ⚠️ Partially real | Boundary policy ($P(B_i{=}1)$, hard $L_{\max}$, soft $\tau_B/L_{\min}$) is correctly implemented in `segment.py`. **Bug:** `target_lengths` ($T_i/L_i$, the paper's adaptive-budget contribution, §3.3) is computed and then never used — dropped as a dead variable. The "Full STP" arm today only exercises the boundary policy, not the budget policy. `variants.py` correctly implements the 6-arm parameter-restriction table via one shared code path — this part matches the calibration methodology exactly. |
| `context/` | `horizon.py`, `reference_signal.py`, `selective_propagation.py` | ✅ Solid (untested at scale) | Implements $H_i$, $R_i$ (coref/abbrev/lexical-cue composite), and $\text{Rel}(c_j,c_i)$ ranking + $\tau_R$ filtering per spec. Only exercised on the 2-doc toy corpus so far — real behavior on long documents is unverified. |
| `indexing/` | `qdrant_store.py` | ✅ Solid | Real `qdrant-client` usage, one collection per arm, graceful in-memory (`:memory:`) fallback when no Docker Qdrant is reachable. No issues found. |
| `retrieval/` | `query_router.py`, `stage1_dense.py` | ⚠️ Incomplete | `query_router.py` is a **regex heuristic**, not the fine-tuned DistilBERT classifier from the tech stack — and separately, it's instantiated in `pipeline.py` but `router.route(...)` is **never called**; it has zero effect on retrieval. `stage1_dense.py` (single-stage dense retrieval) is fine. **Missing entirely:** `stage2_openie.py` and `fusion.py` — the two-stage multi-hop evidence-fusion design from the tech stack doesn't exist in code. |
| `generation/` | `answer.py` | ⚠️ Not functional as run | Correct prompt-assembly logic, but depends on a local Ollama daemon (`phi3:mini`) that was **not running** when `results/*.json` was generated — every predicted answer is literally the string `"[Ollama Offline: ...]"`. This is why EM/F1 are `0.0` in every single result file. |
| `eval/` | `retrieval_metrics.py`, `answer_metrics.py`, `efficiency_metrics.py`, `ragas_eval.py`, `results_schema.py`, `compilation.py` | ⚠️ Structurally good, functionally gapped | Recall@k/MRR/EM/F1/efficiency implementations look correct. **`ragas_eval.py` is never imported or called from `pipeline.py` at all** — the `ragas_*` fields in every result are just the `ArmResult` dataclass defaults, not measured values. `configs/eval.yaml` has `compute_ragas: false`, so this is a known, not accidental, gap — but it means "0.0 faithfulness" in the JSON is currently indistinguishable from "never computed," which is exactly the ambiguity the methodology plan's no-fabrication rule exists to prevent. |
| `tracking/` | `mlflow_utils.py` | ⚠️ Wired but disconnected | `MLflowTracker.run()` context manager exists and is called in `pipeline.py`, but `ArmResult` (with its default `mlflow_run_id=""`) is constructed and **written to disk before** the MLflow run starts — the real run ID is never attached back to the JSON. Every result file has `"mlflow_run_id": ""`, breaking the provenance requirement from the methodology plan. |
| `experiments/pipeline.py` | — | ⚠️ Orchestration works, evaluation is invalid | End-to-end orchestration (ingest → STP → chunk → index → retrieve → generate → score → write) runs and produces valid JSON schema. **Critical bug:** when no `qa_pairs[].gold_chunk_keywords` matches a chunk, the code falls back to treating **every chunk in the document as gold** (`matching_cids = doc_chunk_set`). On a 1–2-document corpus this makes Recall@1/5/10 and MRR trivially `1.0` for all six arms — the evaluation is not discriminating between systems at all right now. |
| `data/splits/` | `dev.json`, `test.json` | 🔴 Not the paper's corpus | 2 synthetic documents in `dev.json`, a similarly tiny `test.json` — clearly a smoke-test fixture, not HotpotQA (the paper's primary corpus) or CUAD. Every result in `results/` was computed on this toy data. |
| `configs/stp_params.yaml` | — | 🔴 Not actually calibrated | Presented as "frozen" $\Theta$, but the Optuna-based dev-split search from the methodology plan (§6.2) hasn't been run — `TODO.md` item 2.3 is unchecked, and there's no calibration notebook output or SHA-256 checksum recorded alongside it, despite `compute_file_sha256()` already existing (unused) in `pipeline.py`. These look like reasonable hand-set priors, not fit values. |
| `notebooks/00`–`10`, `99` | — | ⚠️ Never executed | All 12 notebooks exist with the correct structure, but **every cell has `execution_count: null` and no outputs** — none have actually been run. The `results/*.json` files were produced by `scripts/run_all_experiments.py` instead. The notebook-driven workflow you asked for isn't the one that actually generated any results yet. |
| `tests/` | 6 files | ❓ Unverified | Reasonable coverage by filename (`test_stp_math.py`, `test_segmentation.py`, `test_context_propagation.py`, `test_eval_schema.py`, `test_pipeline_and_compilation.py`, `test_end_to_end_smoke.py`). Could not execute `pytest` in this review sandbox (dependency install hit a disk-space wall) — status is unverified, not confirmed broken. |
| `TODO.md` | — | 🔴 Stale | Every checkbox across all 5 phases is unchecked, despite ~2,400 lines of largely-working code and 18 result files already existing. Not a reliable source of truth for "what's left" — several "unchecked" items are substantially done, and several genuinely-missing items (calibration, RAGAS wiring, real corpus) aren't flagged distinctly from cosmetic ones. |
| `STP_RAG_Overleaf_Package/` | `.tex`, `references.bib`, `table2_generated.tex` | ℹ️ Not reviewed for content | Present and structurally intact; `table2_generated.tex` presumably needs regenerating once real results exist — not inspected in depth this pass. |

---

## Priority Ranking for What to Fix

**P0 — blocks any trustworthy number in Table 2:**
1. Real corpus (HotpotQA dev/test subset, or CUAD for the legal extension) replacing the 2-doc smoke fixture.
2. Fix the gold-chunk fallback in `pipeline.py` so Recall@k/MRR can actually discriminate between arms.
3. Get the generator actually producing answers (Ollama running + model pulled, or swap to an API-based generator) so EM/F1 stop being uniformly 0.
4. Wire `ragas_eval.evaluate_ragas(...)` into `pipeline.py` and set `compute_ragas: true`, with a real judge-LLM key configured.
5. Fix `mlflow_run_id` provenance ordering (attach the real run ID to `ArmResult` before writing JSON).
6. Run the actual Optuna calibration (§6.2) to produce a genuinely fit `stp_params.yaml`, with its SHA-256 checksum recorded and asserted at the top of every ablation run.
7. Wire $T_i/L_i$ (the adaptive budget) into `segment_document`'s actual splitting decision, not just compute-and-discard.

**P1 — scope gaps worth a deliberate decision, not silent omission:**
8. Two-stage retrieval (OpenIE + fusion) — implement, or formally descope multi-hop from this iteration.
9. Either wire `QueryRouter` into the retrieval path for real, or remove it until it does something.
10. Decide and document the embedding model (BGE-M3 vs. the currently-configured bge-small) — fine either way, but should be a stated choice, not a silent config default.

**P2 — hygiene:**
11. Re-sync `TODO.md` with actual repo state.
12. Either execute the notebooks end-to-end (so they're the real source of `results/`) or update the README to reflect that `scripts/run_all_experiments.py` is the actual entry point.

---

**Next:** the prompt below is written to hand to an agentic coding tool (Antigravity) to execute P0 and P1 in one pass, with explicit acceptance criteria per item so it can self-verify instead of declaring victory on vibes.
