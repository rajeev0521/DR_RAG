# STP-RAG: Methodological Implementation Plan
### Formulas, Algorithms, and Calibration Procedure (No Code)

This document specifies *how* to implement the method section of the STP-RAG paper precisely enough to build from — every quantity the paper defines, the quantities it leaves implicit (and must be fixed before implementation), the full algorithmic control flow, and the calibration/evaluation procedure that turns Table 1 into Table 2. It intentionally does not deviate from the paper's method; it fills in implementation-level detail (normalization procedure, structural score, reference signal, ranking function, calibration search) that the paper describes at the level of "must be calibrated" or "must be selected on a validation split" without giving the exact procedure.

---

## 1. Notation and Problem Formulation

A document is an ordered sequence of structure-aware units:
$$
D = \{u_1, u_2, \ldots, u_n\}.
$$

Each unit $u_i$ is one sentence, clause, paragraph, or parser-defined segment (granularity is fixed **per corpus**, never mixed within a comparison). Each unit has:

- an embedding $e_i = E(u_i)$, $e_i$ normalized to unit norm before any distance computation,
- a token position $p_i$ (cumulative token count up to and including $u_i$),
- a metadata tuple $s_i$ recording structural context: `follows_heading`, `follows_list_item`, `follows_clause_marker` (e.g. "however," "except," "provided that"), `is_paragraph_start`.

The output is a set of contextual chunks $C = \{c_1, \ldots, c_m\}$, each eventually represented in indexed form as $\tilde c_i = c_i \Vert \text{SelectedContext}_i$.

**Fixed implementation decision required before any run:** granularity (sentence vs. clause vs. paragraph) and the tokenizer used for $p_i$. Both must be logged as run metadata, since the paper explicitly flags cross-granularity comparison as invalid.

---

## 2. Semantic Transition Profile (STP)

### 2.1 Semantic Displacement

$$
d_i = 1 - \cos(e_{i-1}, e_i), \quad i \geq 2.
$$

$d_i \in [0, 2]$ in principle, but bounded in $[0,1]$ in practice for embeddings that occupy a narrow cone (typical of sentence encoders); no clipping is applied — the raw value is passed forward.

### 2.2 Normalized Progression

The paper leaves $\Delta p_i$ defined only as "normalized progression between units, measured using token positions." The implementation-level definition:
$$
\Delta p_i = \frac{p_i - p_{i-1}}{N},
$$
where $N$ is the total token count of the document. This makes $\Delta p_i$ a *document-relative* progression (dimensionless, sums to 1 over the document), so $v_i$ is comparable across documents of different lengths — an absolute-token-count denominator would instead bias $v_i$ toward long documents.

### 2.3 Semantic Velocity

$$
v_i = \frac{d_i}{\Delta p_i + \epsilon}, \qquad \epsilon = 10^{-6} \text{ (fixed constant, not tuned)}.
$$

$\epsilon$ is set once, globally, and excluded from the calibration search — it exists only to prevent division by zero on zero-length units and should not materially affect $v_i$ for any real unit.

### 2.4 Semantic Acceleration

$$
a_i = v_i - v_{i-1}, \quad i \geq 3.
$$

Sign convention: $a_i > 0$ means the transition rate is increasing (approaching or inside a shift); $a_i < 0$ means the document is stabilizing after a shift. This sign asymmetry is used deliberately in the budget policy (§3.3).

### 2.5 Semantic Volatility

$$
\sigma_i = \sqrt{\frac{1}{w}\sum_{j=i-w+1}^{i}(v_j - \bar v_i)^2}, \qquad \bar v_i = \frac{1}{w}\sum_{j=i-w+1}^{i} v_j,
$$
computed over a trailing window of size $w$. $w$ is a calibrated hyperparameter (§6), not fixed a priori — the paper does not commit to a value, and $w$ should be searched jointly with the boundary/budget coefficients since it changes the meaning of "local" volatility.

**Boundary condition:** for $i < w$, compute $\sigma_i$ over the available prefix ($j = 1$ to $i$) rather than leaving it undefined; document this in the run log since it slightly understates volatility for the first $w$ units of every document.

### 2.6 Structural Score

The paper references $S_i$ as "a scalar structural-boundary score" without giving its form. Implementation-level definition, a bounded weighted sum of binary/graded structural cues:
$$
S_i = \eta_1 \mathbb{1}[\text{follows\_heading}_i] + \eta_2 \mathbb{1}[\text{follows\_list\_item}_i] + \eta_3 \mathbb{1}[\text{is\_paragraph\_start}_i] - \eta_4 \mathbb{1}[\text{follows\_clause\_marker}_i],
$$
with $\eta_1, \ldots, \eta_4 \geq 0$ fixed by the same calibration procedure as the boundary weights (§6), and the clause-marker term subtracted because a clause marker ("however," "provided that") signals continuation of an argument, i.e. evidence *against* a boundary, even when $v_i$ is locally elevated. $S_i$ is normalized to $[0,1]$ by min–max scaling fit on the dev split, consistent with the treatment of $v, a, \sigma$.

### 2.7 Robust Normalization

For each raw signal $x \in \{v, a, \sigma\}$, robust (outlier-resistant) normalization is applied because semantic distance distributions are heavy-tailed:
$$
\hat x_i = \frac{x_i - \operatorname{median}(x)}{\operatorname{IQR}(x) + \delta}, \qquad \delta = 10^{-6},
$$
where $\operatorname{median}(x)$ and $\operatorname{IQR}(x)$ (interquartile range, $Q_3 - Q_1$) are computed **once, over the dev split only**, and frozen. Test-split units are normalized using the frozen dev statistics — never re-fit on test data. This is the single most important procedural rule in the whole pipeline: refitting normalization on test data silently leaks test-distribution information into every downstream decision.

### 2.8 The Profile

$$
\mathrm{STP}_i = [\hat v_i, \hat a_i, \hat \sigma_i, S_i].
$$

---

## 3. Boundary and Granularity Policy

### 3.1 Boundary Probability

$$
P(B_i = 1) = \operatorname{sigmoid}\big(w_v \hat v_i + w_a \hat a_i + w_\sigma \hat \sigma_i + w_s S_i + b\big).
$$

### 3.2 Boundary Acceptance Rule

A boundary after unit $i$ is accepted iff **both**:
$$
P(B_i = 1) > \tau_B \quad \text{and} \quad \text{len}(c_{\text{current}}) \geq L_{\min}.
$$
If the current chunk reaches $L_{\max}$ tokens before either condition is met, a boundary is forced regardless of $P(B_i)$ — this is a hard ceiling, not a soft preference, and exists purely to guarantee indexability in stable regions where $P(B_i)$ may never cross $\tau_B$.

### 3.3 Local Target Budget

$$
T_i = \alpha \hat v_i + \beta \max(\hat a_i, 0) + \gamma \hat \sigma_i.
$$

Only the *positive* part of acceleration enters $T_i$ — a decelerating transition rate ($\hat a_i < 0$) is evidence the document is settling into a stable region and should not, by itself, push toward a smaller budget.

$$
L_i = \operatorname{clip}\big(L_{\max} - (L_{\max} - L_{\min})\,T_i,\; L_{\min},\; L_{\max}\big).
$$

Interpretation: $T_i \to 0$ (stable region) $\Rightarrow L_i \to L_{\max}$ (large chunks); $T_i \to 1$ (high transition activity) $\Rightarrow L_i \to L_{\min}$ (fine-grained chunks). $T_i$ itself is **not** bounded to $[0,1]$ by construction (it's a weighted sum of normalized-but-unbounded signals), so the `clip` on $L_i$ is what actually enforces the budget range — $T_i$ can exceed 1 or go negative and the clip absorbs it.

### 3.4 Segmentation Algorithm

**Algorithm A — `Segment(U, P, Θ, L_min, L_max)`**

```
Input:  units U = {u_1,...,u_n}, profile P = {STP_1,...,STP_n},
        parameters Θ = {w_v,w_a,w_σ,w_s,b,τ_B,α,β,γ}, L_min, L_max
Output: chunk boundary set

1. current_length ← 0
2. boundaries ← {}
3. for i = 2 to n:
4.     compute P(B_i=1) via §3.1 using STP_i, Θ
5.     compute T_i, L_i via §3.3 using STP_i, Θ  (L_i informs early-stop heuristics
       but does not itself trigger a split — only §3.2's rule does)
6.     current_length ← current_length + tokens(u_i)
7.     if current_length ≥ L_max:
8.         accept boundary at i  (forced)
9.         current_length ← 0
10.    else if P(B_i=1) > τ_B and current_length ≥ L_min:
11.        accept boundary at i
12.        current_length ← 0
13. append final boundary at n if not already closed
14. return boundaries → induces chunks {c_1,...,c_m}
```

Note that $L_i$ is computed at every step but is advisory: the paper's binding constraints are $L_{\min}/L_{\max}$ and $\tau_B$. $L_i$'s practical role is as a **diagnostic** (compare realized chunk lengths against the locally predicted target as a calibration sanity check) and, optionally, as a secondary early-boundary trigger in dense-transition regions — but the paper's Algorithm 1 does not require this second use, so treat it as an extension, not baseline behavior, and keep it out of the "Full STP" arm's core comparison unless explicitly ablated as its own variant.

---

## 4. Selective Context Propagation

### 4.1 Context Horizon

$$
H_i = \operatorname{clip}\big(H_{\min} + \lambda_1 \hat \sigma_i + \lambda_2 \max(\hat a_i, 0) + \lambda_3 R_i,\; H_{\min},\; H_{\max}\big).
$$

$H_i$ is measured in **number of preceding chunks** (not units) considered as candidates, consistent with the paper's phrase "candidate predecessors in the horizon."

### 4.2 Reference/Dependency Signal

The paper leaves $R_i$ as "a reference/dependency signal" without a formula. Implementation-level definition, a chunk-level score combining three sub-signals, each in $[0,1]$:

$$
R_i = \max\big(R_i^{\text{coref}},\; R_i^{\text{abbrev}},\; R_i^{\text{lexcue}}\big),
$$

- $R_i^{\text{coref}} = 1$ if $c_i$ contains an unresolved anaphoric reference (a pronoun or demonstrative — "this condition," "the above," "it") whose antecedent is not inside $c_i$ itself; else $0$.
- $R_i^{\text{abbrev}} = 1$ if $c_i$ uses an abbreviation or defined term not itself defined within $c_i$; else $0$.
- $R_i^{\text{lexcue}} = 1$ if $c_i$ opens with a discourse connective presupposing prior content ("therefore," "as a result," "similarly," "in contrast"); else $0$.

The max (rather than sum) is deliberate: any one of these is sufficient evidence that context is needed; they should not compound the horizon size beyond what a single strong signal already justifies.

### 4.3 Candidate Ranking

For each candidate predecessor chunk $c_j$, $j \in [i-H_i, i-1]$, a composite relevance score:
$$
\text{Rel}(c_j, c_i) = \mu_1 \cos(\bar e_j, \bar e_i) + \mu_2 \,\text{DepMatch}(c_j, c_i) + \mu_3 \,\text{RefResolve}(c_j, c_i),
$$
where $\bar e_j$ is the mean (or max-pooled) embedding of $c_j$'s constituent units, $\text{DepMatch}$ scores shared syntactic dependents/entities between $c_j$ and $c_i$ (e.g. normalized overlap of named entities and key noun phrases), and $\text{RefResolve} \in \{0,1\}$ indicates that $c_j$ contains the antecedent/definition that triggered $R_i > 0$ in §4.2. $\mu_1+\mu_2+\mu_3$ need not sum to 1; the scale of $\text{Rel}$ is only meaningful relative to the threshold $\tau_R$ below, which is calibrated jointly.

### 4.4 Selection Rule

$$
\text{SelectedContext}_i = \{\, c_j : j \in [i-H_i, i-1],\; \text{Rel}(c_j, c_i) > \tau_R \,\}.
$$

Only chunks clearing $\tau_R$ are attached — $H_i$ sets the *candidate pool size*, $\tau_R$ decides *actual inclusion*. This two-stage design (widen, then filter) is what keeps the context genuinely selective instead of just "attach the last $H_i$ chunks."

### 4.5 Augmentation

$$
\tilde c_i = c_i \,\Vert\, \text{SelectedContext}_i,
$$
concatenated in chunk order, with the raw $c_i$ and the identities of any attached predecessors retained separately as provenance metadata (required for later evidence inspection and for the "Full STP" vs. "Full STP + context" ablation to be auditable).

---

## 5. Master Algorithm (Full Pipeline)

**Algorithm 1 — `STP-Guided Contextual Chunking`** (expanded)

```
Require: Document D, embedding model E, frozen parameters Θ
Ensure:  Contextual chunks {c̃_i} with provenance

 1. U   ← ExtractStructureAwareUnits(D)            (§1; fixed granularity)
 2. E_U ← Embed(U, E)                                (§1; unit-normalized)
 3. S   ← StructureFeatures(U)                        (§2.6, raw flags)
 4. for i = 2 to n: compute d_i, v_i                  (§2.1–2.2)
 5. for i = 3 to n: compute a_i                        (§2.3)
 6. for i = w to n: compute σ_i                        (§2.4; prefix-window for i<w)
 7. Fit or load frozen normalization stats             (§2.7 — LOAD, not fit, at inference time)
 8. Compute hat-values, STP_i = [v̂_i, â_i, σ̂_i, S_i]   (§2.8)
 9. C ← Segment(U, STP, Θ, L_min, L_max)                (§3.4, Algorithm A)
10. for each c_i in C:
11.     H_i ← ContextHorizon(STP_i, Θ)                  (§4.1)
12.     candidates ← chunks in [i-H_i, i-1]
13.     X_i ← { c_j ∈ candidates : Rel(c_j, c_i) > τ_R } (§4.3–4.4)
14.     c̃_i ← Augment(c_i, X_i)                          (§4.5)
15. return {c̃_i} with raw-text + attached-predecessor provenance
```

---

## 6. Calibration Methodology (Fitting Θ)

The paper requires that "coefficients and thresholds should be selected on a held-out validation split," without prescribing the search procedure. The methodology below fills that gap.

### 6.1 Parameter Set

$$
\Theta = \{\, w_v, w_a, w_\sigma, w_s, b,\; \tau_B,\; L_{\min}, L_{\max},\; \alpha, \beta, \gamma,\; w \text{ (volatility window)},\; \eta_1..\eta_4,\; H_{\min}, H_{\max}, \lambda_1, \lambda_2, \lambda_3,\; \mu_1, \mu_2, \mu_3, \tau_R \,\}.
$$

### 6.2 Two-Stage Calibration Procedure

**Stage A — Normalization fit (unsupervised, deterministic).** Compute median/IQR for $v, a, \sigma$ over the dev split only (§2.7). No search — this is a closed-form statistic, computed once.

**Stage B — Downstream-metric-driven search (the boundary, budget, horizon, and ranking coefficients).** Because the paper does not supply ground-truth topic-boundary labels, coefficients are calibrated by directly optimizing a dev-split downstream retrieval metric (Recall@5, chosen as the primary calibration objective since it is the least sensitive to generator variance), using:

1. **Coarse grid search** over $\tau_B \in \{0.3, 0.4, \ldots, 0.7\}$, $w \in \{3, 5, 8\}$, and $L_{\min}/L_{\max}$ pairs drawn from corpus token-length statistics (e.g. 10th/90th percentile of natural paragraph lengths), holding $w_v, w_a, w_\sigma, w_s$ at a neutral prior (equal weight, normalized to sum to 1).
2. **Bayesian optimization** (e.g. Gaussian-process- or tree-structured-Parzen-based) refinement of the continuous weights $w_v, w_a, w_\sigma, w_s, b, \alpha, \beta, \gamma, \eta_{1..4}, \lambda_{1..3}, \mu_{1..3}, \tau_R$ within bounds centered on the coarse-search optimum, budgeted to a fixed number of dev-split evaluations (e.g. 50–100 trials) to keep the search itself from overfitting the dev split.
3. **Selection rule:** the trial maximizing dev-split Recall@5 is frozen; ties broken by the trial with the lower dev-split retrieval latency (a soft preference for simpler configurations among near-equal performers).

**Stage C — Freeze.** All of $\Theta$ plus the normalization statistics are written to a fixed configuration and are not touched again until every test-split run in the paper's final table has been executed and logged.

### 6.3 Per-Ablation-Arm Reduction

Each row of Table 1 is a **restriction** of the same $\Theta$, not a separately calibrated model, so that differences in Table 2 are attributable to which signals are active rather than to independently re-tuned parameters:

| Arm | Restriction applied to §3.1/§3.3/§4.1 |
|---|---|
| Fixed-size | $w_v=w_a=w_\sigma=w_s=0$; fixed $L_i = L_{\max}$; $H_i = 0$ |
| Similarity-threshold | Only $d_i$ (pre-velocity distance) drives a single-threshold rule; $a,\sigma,S$ excluded |
| Velocity-only | $w_a = w_\sigma = 0$; $\beta = \gamma = 0$; $\lambda_1=\lambda_2=0$ |
| Velocity + acceleration | $w_\sigma = 0$; $\gamma = 0$; $\lambda_1 = 0$ |
| Full STP | all of $w_v,w_a,w_\sigma,w_s$ and $\alpha,\beta,\gamma$ active; $H_i$ fixed at $H_{\min}$ (no selective context) |
| Full STP + selective context | full $\Theta$, including $H_i$ and context selection (§4) |

Each restricted arm still uses Stage B's search, but confined to its active parameter subset, so "Fixed-size" isn't handicapped by an untuned $L_{\max}$, for instance.

---

## 7. Evaluation Formulas

### 7.1 Retrieval Metrics

$$
\text{Recall@}k = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}\big[\exists\, r \leq k : \text{chunk}_r(q) \text{ is answer-bearing}\big],
$$
$$
\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q},
$$
where $\text{rank}_q$ is the position of the first answer-bearing chunk retrieved for query $q$ (defined as $\infty$, contributing 0, if none appears in the retrieved set).

### 7.2 Answer-Quality Metrics

$$
\text{EM} = \frac{1}{|Q|}\sum_{q} \mathbb{1}[\hat a_q = a_q^*] \quad \text{(after standard normalization: lowercase, strip punctuation/articles)},
$$
$$
F1 = \frac{1}{|Q|}\sum_q \frac{2 \cdot P_q \cdot R_q}{P_q + R_q}, \quad P_q = \frac{|\text{tokens}(\hat a_q) \cap \text{tokens}(a_q^*)|}{|\text{tokens}(\hat a_q)|}, \; R_q = \frac{|\text{tokens}(\hat a_q) \cap \text{tokens}(a_q^*)|}{|\text{tokens}(a_q^*)|}.
$$

### 7.3 Efficiency Metrics

$$
\overline{\text{chunks/doc}} = \frac{1}{|D_{\text{corpus}}|}\sum_{D} |C_D|, \qquad \overline{\text{tokens/chunk}} = \frac{1}{|C|}\sum_{c \in C} \text{tokens}(c),
$$
plus wall-clock index build time and per-query retrieval latency, both measured under fixed hardware/batch settings identical across all six arms.

### 7.4 RAGAS Metrics (formal definitions used for reporting)

- **Faithfulness** $= \dfrac{\text{\# claims in generated answer supported by retrieved context}}{\text{\# claims in generated answer}}$.
- **Answer relevancy**: mean cosine similarity between the embedding of the original question and embeddings of several questions reverse-generated from the answer.
- **Context precision**: precision of retrieved chunks with respect to ground-truth relevance, computed rank-weighted (relevant chunks ranked earlier score higher).
- **Context recall**: proportion of ground-truth answer-supporting statements attributable to the retrieved context.

### 7.5 Statistical Protocol

For each arm, run $\geq 3$ seeds where any component is stochastic (chunk-boundary ties, sampling in generation, router thresholding at decision boundary); report $\mu \pm \sigma$ per metric. For pairwise arm comparison (e.g. Full STP vs. Velocity-only), use the paired bootstrap:
$$
\hat\Delta = \text{metric}(\text{Arm}_A) - \text{metric}(\text{Arm}_B), \quad \text{CI}_{95\%} = \text{percentile}\big(\{\hat\Delta^{(b)}\}_{b=1}^{B},\, [2.5, 97.5]\big),
$$
resampling query-level pairs with replacement $B$ (e.g. 10,000) times; an effect is reported as significant only if the CI excludes 0.

---

## 8. Reporting Checklist (Ties Back to Table 2)

- [ ] Every populated cell in Table 2 traces to a specific $(\text{arm}, \text{seed}, \text{split})$ run using the frozen $\Theta$ from §6.3.
- [ ] $\mu \pm \sigma$ reported wherever $\geq 3$ seeds were run; single-seed cells explicitly flagged, not silently presented as equivalent.
- [ ] Paired bootstrap CIs reported for each arm-vs.-velocity-only comparison per §7.5, feeding directly into the paper's three closing questions (does each added signal help; is any gain latency/index-cost-driven; do gains vary by document-structure type).
- [ ] Granularity (sentence/clause/paragraph) logged per corpus and never mixed across a single Table 2 row.
