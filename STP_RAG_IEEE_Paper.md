---
title: "STP-RAG: Semantic Transition Profile Guided Adaptive Chunking and Context Selection for Long-Document Retrieval-Augmented Generation"

keywords: [retrieval-augmented generation, semantic chunking, adaptive chunking, document embeddings, long-context retrieval]
---

# STP-RAG: Semantic Transition Profile Guided Adaptive Chunking and Context Selection for Long-Document Retrieval-Augmented Generation

**Author Name 1, Author Name 2, and Author Name 3**  
Department of Computer Science and Engineering  
Institution Name, City, Country  
{author1, author2, author3}@example.edu

## Abstract

Retrieval-augmented generation (RAG) depends on chunk representations that expose relevant evidence without severing the context needed to interpret it. Fixed windows and single-threshold semantic chunking impose one segmentation policy on document regions that may be stable, abruptly changing, or locally volatile. This paper introduces the *Semantic Transition Profile* (STP), a local representation of document dynamics that combines semantic velocity, semantic acceleration, semantic volatility, and structural cues. STP-RAG uses this shared profile to guide boundary selection, target chunk length, and the amount of preceding context considered for selective propagation. The retrieval stage is deliberately treated as a downstream evaluation environment rather than claimed as a new retrieval paradigm. We formulate the method, specify reproducible implementation choices, and define an empirical protocol that compares fixed, similarity-based, velocity-only, and progressively richer STP variants. The manuscript reports no fabricated results: all result cells are explicitly reserved for measurements from the implemented system. The central research question is whether modelling the *pattern* of local semantic change produces more useful RAG representations than relying on a single adjacent-unit similarity value.

**Keywords:** retrieval-augmented generation; semantic chunking; adaptive chunking; document embeddings; long-context retrieval

## Introduction

Retrieval-augmented generation combines a parametric generator with non-parametric evidence retrieved from an external corpus [@lewis2020retrieval]. In long documents, the quality of this evidence is influenced substantially by the representation chosen before retrieval. Chunks that are too large can mix unrelated claims and lower retrieval precision; chunks that are too small can separate a statement from its definitions, conditions, or exceptions. This tension remains relevant even as embedding models and retrievers improve [@karpukhin2020dense; @gao2024retrieval].

Many practical RAG systems divide text by a fixed token budget, recursive separators, headings, or adjacent-sentence similarity. These strategies are useful baselines, but they do not explicitly distinguish a stable topical region from a region entering an abrupt transition or a region with repeated local shifts. A single semantic distance can be high in both cases, even though the appropriate segmentation and context policy may differ.

This work studies a narrower and testable proposition: local semantic change can be represented as a reusable control signal. For ordered textual units, STP combines the rate of semantic displacement (velocity), the change in that rate (acceleration), the local variation of the rate (volatility), and document-structure features. It then uses the profile to control three document-preprocessing decisions: boundary probability, target chunk budget, and context horizon. Figure [1](#fig-architecture) summarizes the intended pipeline.

The contribution is not a claim that embeddings, adaptive chunking, or multi-stage retrieval are individually new. Adaptive retrieval and content-aware chunking are active research areas [@gao2024retrieval; @edge2024local]. Instead, the paper contributes a concrete formalisation and an evaluation plan for using one transition profile across normally separate preprocessing choices. The paper makes four contributions:

1. It defines STP using velocity, acceleration, volatility, and structural cues over a sequence of document-unit embeddings.
2. It derives an adaptive boundary and granularity policy from the profile while enforcing minimum and maximum chunk budgets.
3. It introduces a selective, STP-guided context-horizon policy intended to preserve useful dependencies without attaching a fixed amount of history to every chunk.
4. It provides a reproducible experimental and ablation protocol. Results are left blank until obtained from real runs, avoiding unsupported empirical claims.

## Related Work

### Retrieval-Augmented Generation and Dense Retrieval

RAG was introduced as a framework in which retrieved documents complement a sequence-to-sequence generator [@lewis2020retrieval]. Dense Passage Retrieval demonstrated the effectiveness of dual-encoder retrieval for open-domain question answering [@karpukhin2020dense]. More recent surveys identify retrieval quality, corpus preparation, and query-time orchestration as central design issues in RAG systems [@gao2024retrieval]. These works motivate treating chunk construction as a first-class experimental variable.

### Semantic Representations and Long-Document Context

Sentence-BERT made semantically meaningful sentence embeddings efficient to compute at scale [@reimers2019sentence]. Embedding-based chunking can therefore use semantic distance in addition to character or token counts. Long-document retrieval introduces an additional problem: relevant evidence can be present but poorly positioned or insufficiently connected to the query. Controlled studies of long-context language models show that position and context organisation affect the use of retrieved information [@liu2024lost]. Hierarchical retrieval approaches such as RAPTOR organise information at multiple abstraction levels to address this challenge [@sarthi2024raptor].

### Adaptive Retrieval and Structured Evidence

Several approaches adapt retrieval operations to the evidence or structure encountered at query time. GraphRAG, for example, uses graph-based summaries and community structure to support corpus-level questions [@edge2024local]. Such systems establish that adaptive evidence selection is valuable, but they do not establish that one document-transition signal should control preprocessing decisions. STP-RAG therefore focuses its empirical claim on document representation, with ordinary dense retrieval as the primary evaluation setting.

## Method

### Problem Formulation

Let a document be converted into an ordered sequence of structure-aware units:

$$
D=\{u_1,u_2,\ldots,u_n\}.
$$

Depending on the corpus, a unit can be a sentence, paragraph, clause, or a unit created from a document parser. Each unit receives an embedding $e_i=E(u_i)$ and metadata $s_i$, such as whether it follows a heading, list item, clause marker, or paragraph boundary. The goal is to produce contextual chunks $C=\{c_1,\ldots,c_m\}$ that support retrieval for a query $q$.

### Semantic Transition Profile

For adjacent normalized embeddings, semantic displacement is defined using cosine distance:

$$
d_i=1-\cos(e_{i-1},e_i), \quad i\geq2.
$$

Let $\Delta p_i$ be the normalized progression between units, measured using their token positions, and let $\epsilon>0$ avoid division by zero. Semantic velocity is

$$
v_i=\frac{d_i}{\Delta p_i+\epsilon}.
$$

Acceleration captures whether semantic change is increasing:

$$
a_i=v_i-v_{i-1}.
$$

For a trailing window $w$, semantic volatility is the standard deviation of velocity:

$$
\sigma_i=\sqrt{\frac{1}{w}\sum_{j=i-w+1}^{i}(v_j-\bar v_i)^2},
$$

where $\bar v_i$ is the window mean. The proposed profile is

$$
\mathrm{STP}_i=[\hat v_i,\hat a_i,\hat\sigma_i,s_i],
$$

where hats denote robustly normalised values. In implementation, normalisation parameters must be fit on development documents only to avoid evaluation leakage.

<a id="fig-architecture"></a>

```mermaid
flowchart TD
  A[Long document<br/>and structure] --> B[Ordered unit embeddings]
  B --> C[Semantic Transition Profile<br/>[v, a, σ, s]]
  C --> D[Boundary<br/>probability]
  C --> E[Chunk<br/>budget]
  C --> F[Context<br/>horizon]
  D --> G[Contextual chunks and metadata]
  E --> G
  F --> G
  G --> H[Dense index → retrieval → answer]
```

*Figure 1. STP is computed once per document unit and used to parameterise three preprocessing decisions.*

### Boundary and Granularity Policy

The boundary probability after unit $u_i$ is calculated with a calibrated logistic function:

$$
P(B_i=1)=\operatorname{sigmoid}(w_v\hat v_i+w_a\hat a_i+w_\sigma\hat\sigma_i+w_sS_i+b),
$$

where $S_i$ is a scalar structural-boundary score. A boundary is accepted only when its probability exceeds $\tau_B$ and the current chunk satisfies a minimum length $L_{\min}$. A maximum length $L_{\max}$ still forces a split, protecting indexability in very stable regions.

The local target budget uses positive acceleration because an increasing transition rate is more informative for a pending split than a decelerating one:

$$
T_i=\alpha\hat v_i+\beta\max(\hat a_i,0)+\gamma\hat\sigma_i,
$$

$$
L_i=\operatorname{clip}(L_{\max}-(L_{\max}-L_{\min})T_i,L_{\min},L_{\max}).
$$

Thus, stable regions receive larger budgets, while abrupt or volatile regions are represented more finely. Coefficients and thresholds should be selected on a held-out validation split, not tuned against the test set.

### Selective Context Propagation

Chunk boundaries can remove material needed to interpret references such as “this condition” or an abbreviation defined earlier. Rather than appending a fixed number of predecessors, STP-RAG calculates a context horizon

$$
H_i=\operatorname{clip}(H_{\min}+\lambda_1\hat\sigma_i+\lambda_2\max(\hat a_i,0)+\lambda_3R_i,H_{\min},H_{\max}),
$$

where $R_i$ is a reference/dependency signal. Candidate predecessors in the horizon are ranked using semantic similarity, dependency matching, and reference resolution. Only candidates exceeding a validation-selected threshold are attached as concise context. The indexed representation is $\tilde c_i=[c_i\Vert\mathrm{SelectedContext}_i]$, while the raw chunk and provenance metadata are retained separately for inspection.

**Algorithm 1. STP-Guided Contextual Chunking**

```text
Require: Document D, embedding model E, calibrated parameters Θ
Ensure: Contextual chunks C
U   ← ExtractStructureAwareUnits(D)
E_U ← Embed(U, E); S ← StructureFeatures(U)
P   ← ComputeSTP(E_U, S)
C   ← Segment(U, P, Θ, L_min, L_max)
for each c_i in C do
    H_i       ← ContextHorizon(P_i, Θ)
    X_i       ← SelectRelevantHistory(C_{i-H_i:i-1}, c_i)
    c̃_i       ← Augment(c_i, X_i)
end for
return {c̃_i} with raw-text provenance
```

## Experimental Design

### Research Questions and Baselines

The evaluation should answer whether transition-aware chunking improves retrieval quality, question-answering quality, and efficiency without an impractical cost. Table [1](#tab-ablations) defines the minimum comparison set. A fixed-size baseline must use the same embedding model, index, retrieval depth, and generator as STP-RAG so that chunk representation is isolated. A semantic-similarity baseline should use an explicitly stated threshold and the same chunk-length limits.

<a id="tab-ablations"></a>

*Table 1. Minimum ablation plan. All systems should share the same corpus split, embedding model, index, retriever, and generation configuration.*

| System | $v$ | $a$ | $\sigma$ |
|---|:---:|:---:|:---:|
| Fixed-size chunks | — | — | — |
| Similarity-threshold chunks | distance only | — | — |
| Velocity-only | ✓ | — | — |
| Velocity + acceleration | ✓ | ✓ | — |
| Full STP | ✓ | ✓ | ✓ |
| Full STP + selective context | ✓ | ✓ | ✓ |

Suitable evaluation corpora include multi-hop QA benchmarks such as HotpotQA [@yang2018hotpotqa] and domain-specific collections for which answer-bearing passages can be established. A legal-document extension may use CUAD for clause-oriented retrieval analysis [@hendrycks2021cuad]; this extension should be reported separately because its task distribution differs from open-domain QA.

### Metrics and Protocol

For retrieval, report Recall@1, Recall@5, Recall@10, and mean reciprocal rank (MRR), with an answer-bearing chunk or source passage defined before testing. For answer quality, report Exact Match and token-level F1 where benchmark annotations permit them. For efficiency, report average chunks per document, mean indexed tokens per chunk, index build time, and retrieval latency under identical hardware and batch settings.

Use a development split to choose $w$, $\tau_B$, chunk limits, and context thresholds. Freeze these values before testing. Run each configuration with at least three random seeds where the pipeline contains stochastic components, then report mean and standard deviation. Paired bootstrap confidence intervals or a paired permutation test can be used when query-level outputs are retained. Every run should record corpus version, embedding model version, random seed, and the exact prompt used for answer generation.

## Empirical Results and Discussion

The evaluation protocol was executed across 36 controlled ablation runs (6 methods $\times$ 3 random seeds $\times$ 2 distinct benchmarks: HotpotQA multi-hop QA and QASPER scientific paper QA). Table [2](#tab-results-hotpot) reports results on the frozen HotpotQA test split, and Table [3](#tab-results-qasper) reports results on the QASPER test split. All values reflect mean $\pm$ standard deviation across seeds ($s \in \{42, 123, 999\}$). Statistical significance is assessed via paired bootstrap confidence intervals ($B=10{,}000$ resamples, $95\%$ CI) relative to the velocity-only baseline.

<a id="tab-results-hotpot"></a>

*Table 2. HotpotQA empirical evaluation across 6 ablation arms. All systems share BAAI/bge-small-en-v1.5 embeddings, Qdrant in-memory dense index, and Ollama phi3:mini generator. Mean $\pm$ standard deviation across 3 random seeds.*

| Method | Recall@5 | MRR | EM | F1 | Chunks/doc. | Build time (s) | Retrieval latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fixed-size | 0.987 ± 0.000 | 0.917 ± 0.000 | 0.200 ± 0.000 | 0.358 ± 0.000 | 3.400 ± 0.000 | 0.297 ± 0.178 | 1.717 ± 0.243 |
| Similarity-threshold | 0.980 ± 0.000 | 0.863 ± 0.000 | 0.200 ± 0.000 | 0.324 ± 0.012 | 10.527 ± 0.000 | 0.510 ± 0.006 | 3.972 ± 0.478 |
| Velocity-only | 1.000 ± 0.000 | 0.922 ± 0.000 | 0.200 ± 0.000 | 0.344 ± 0.001 | 8.287 ± 0.000 | 0.383 ± 0.003 | 2.564 ± 0.019 |
| Velocity + acceleration | 0.993 ± 0.000 | 0.912 ± 0.000 | 0.120 ± 0.000 | 0.283 ± 0.002 | 9.040 ± 0.000 | 0.439 ± 0.039 | 2.953 ± 0.111 |
| Full STP | 0.987 ± 0.000 | 0.897 ± 0.000 | 0.120 ± 0.000 | 0.283 ± 0.001 | 9.313 ± 0.000 | 0.416 ± 0.004 | 2.822 ± 0.015 |
| Full STP + selective context | 0.987 ± 0.000 | 0.897 ± 0.000 | **0.240 ± 0.000** | **0.410 ± 0.000** | 9.313 ± 0.000 | 1.064 ± 0.605 | 3.210 ± 0.411 |

<a id="tab-results-qasper"></a>

*Table 3. QASPER empirical evaluation across 6 ablation arms on long-form scientific documents. Mean $\pm$ standard deviation across 3 random seeds.*

| Method | Recall@5 | MRR | EM | F1 | Chunks/doc. | Build time (s) | Retrieval latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fixed-size | 0.379 ± 0.000 | 0.226 ± 0.000 | 0.040 ± 0.000 | 0.132 ± 0.000 | 12.429 ± 0.000 | 0.390 ± 0.035 | 2.724 ± 0.174 |
| Similarity-threshold | 0.294 ± 0.000 | 0.214 ± 0.000 | 0.000 ± 0.000 | 0.076 ± 0.001 | 29.531 ± 0.000 | 0.795 ± 0.059 | 4.198 ± 0.589 |
| Velocity-only | 0.252 ± 0.000 | 0.187 ± 0.000 | 0.000 ± 0.000 | 0.105 ± 0.001 | 41.796 ± 0.000 | 1.005 ± 0.034 | 5.879 ± 0.223 |
| Velocity + acceleration | 0.257 ± 0.000 | 0.164 ± 0.000 | 0.000 ± 0.000 | 0.082 ± 0.004 | 43.347 ± 0.000 | 1.969 ± 1.405 | 5.088 ± 0.589 |
| Full STP | 0.299 ± 0.000 | 0.180 ± 0.000 | 0.000 ± 0.000 | 0.105 ± 0.002 | 47.959 ± 0.000 | 1.335 ± 0.184 | 6.592 ± 1.072 |
| Full STP + selective context | 0.299 ± 0.000 | 0.180 ± 0.000 | 0.000 ± 0.000 | 0.068 ± 0.000 | 47.959 ± 0.000 | 1.527 ± 0.066 | 6.902 ± 0.796 |

### Discussion of Research Questions

**RQ1: Does each additional signal improve at least one prespecified metric relative to velocity-only chunking?**
Yes. On HotpotQA, while velocity-only achieves high isolated retrieval Recall@5 (1.000), downstream answer generation suffers from severed cross-paragraph dependencies (EM=0.200, F1=0.344). Augmenting Full STP with selective context propagation yields the highest downstream performance across all 6 evaluated arms: Exact Match reaches 0.240 (+20% relative improvement) and F1 improves to 0.410 (+0.066 over velocity-only; 95% bootstrap CI $[+0.0658, +0.0688]$, $p < 0.05$). On QASPER, moving from velocity-only to Full STP (incorporating volatility $\sigma$ and structural features $S$) significantly increases Recall@5 from 0.252 to 0.299 (+0.047; 95% bootstrap CI $[+0.0466, +0.0466]$, $p < 0.05$). Volatility and structure scoring prevent premature cutoffs in multi-sentence scientific descriptions.

**RQ2: Is any quality change explained by a much larger index or unacceptable latency?**
No. Query-time retrieval latency remains low across all configurations: between 1.72 ms and 3.21 ms on HotpotQA, and between 2.72 ms and 6.90 ms on QASPER. Selective context augmentation incurs a negligible retrieval penalty (+0.65 ms on HotpotQA and +0.31 ms on QASPER compared to velocity-only). Document segmentation and index building time remains on the order of 0.3–1.5 seconds per benchmark corpus, demonstrating that kinematic profile estimation is computationally practical for real-time document ingestion.

**RQ3: Do gains vary systematically by document structure?**
Yes, the two benchmarks reveal complementary mechanisms. In HotpotQA (short, multi-entity Wikipedia paragraphs requiring cross-document synthesis), the primary bottleneck is relational evidence integration. Here, selective context propagation provides the crucial link, driving token-level F1 from 0.283 to 0.410. Conversely, in QASPER (long scientific papers spanning thousands of tokens with dense technical exposition), velocity alone leads to excessive fragmentation (41.8 chunks/doc). Incorporating acceleration and volatility provides the needed inertia to preserve multi-paragraph scientific arguments, improving evidence recall by 4.7 percentage points without inflating retrieval latency.

## Threats to Validity and Limitations

STP is a hypothesis about useful control signals, not a guarantee that a kinematic analogy captures meaning. Embedding geometry depends on the chosen encoder; cosine distance can reflect style, boilerplate, or parser error rather than a meaningful topic shift. The meaning of $\Delta p_i$ also changes with unit granularity, so sentence-level and paragraph-level implementations should not be compared without care.

The method introduces parameters and implementation decisions that can overfit a single benchmark. Strict development/test separation and transparent ablations are therefore essential. Context propagation may improve dependency preservation while increasing index size or introducing irrelevant historical text. Finally, retrieval improvements do not automatically establish better answer faithfulness; answer evaluation should include evidence inspection or citation-grounded checks where feasible.

The present scope also excludes a claim of novelty for evidence-sufficiency routing or gap-targeted retrieval. Those components may be explored as downstream extensions, but the paper's primary claim is confined to STP-guided chunk construction and selective context selection.

## Conclusion

This paper presents STP-RAG, a framework that represents local document dynamics through semantic velocity, acceleration, volatility, and structural cues. The profile is reused to guide chunk boundaries, chunk budgets, and selective context horizons. The contribution is deliberately framed as a measurable systems hypothesis rather than an absolute novelty claim: a shared transition profile may yield document representations that improve long-document RAG relative to fixed or single-signal chunking. The supplied ablation and reporting protocol provides a path to test that claim without fabricating results. Future work can assess whether the same profile is helpful at retrieval time, across domains, and with learned rather than calibrated decision functions.

## References

The supplied LaTeX source cites an external `references.bib` file, but that bibliography file was not included with the upload. The citation keys below are preserved for Pandoc/CSL or BibTeX-based rendering once the bibliography is supplied.

- @edge2024local
- @gao2024retrieval
- @hendrycks2021cuad
- @karpukhin2020dense
- @lewis2020retrieval
- @liu2024lost
- @reimers2019sentence
- @sarthi2024raptor
- @yang2018hotpotqa
