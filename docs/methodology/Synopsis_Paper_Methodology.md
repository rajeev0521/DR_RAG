# Formal Methodology for Synopsis & Research Paper

---

## 3. Proposed Methodology

The proposed framework, **Adaptive Semantic-Velocity Chunking with Dynamic-Relevant RAG (DR-RAG)**, introduces a computationally efficient workflow for semantic indexing and multi-hop retrieval over complex, structured documents. The methodology proactively addresses critical limitations inherent to naive Retrieval-Augmented Generation (RAG) paradigms—specifically, context fragmentation caused by arbitrary token truncation, and the failure of single-stage cosine similarity to bridge multi-hop reasoning gaps.

The architecture is bifurcated into two primary modules: an intelligent indexing pipeline and a two-stage adaptive retrieval mechanism.

### 3.1. Phase I: Context-Augmented Agentic Chunking (CAAC)

Traditional RAG systems model a document $D$ as a continuous sequence of tokens sub-divided into fixed, non-overlapping or partially overlapping windows $W_1, W_2, \dots, W_n$. This heuristic approach severely neglects domain-specific semantic structures. The CAAC module replaces static windows with dynamically determined semantic boundaries, enriched by historical context.

**3.1.1. SLM-Guided Boundary Detection**
Let $D$ denote a source document. A localized Small Language Model (SLM), parameterized by $\theta_{SLM}$, serves as a parsing agent. The SLM evaluates $D$ and identifies optimal semantic breakpoints $b_1, b_2, \dots, b_k$ based on document-specific heuristics (e.g., logical clauses in legal contracts, or propositional statements in academic literature). This partitioning yields a set of contiguous semantic chunks $C = \{C_1, C_2, \dots, C_k\}$.

**3.1.2. Velocity-Driven Context Augmentation**
To mitigate context decay—wherein deeper document chunks lose connection to their overarching thesis—a rolling abstractive summarization technique is applied. Let $S_{roll}$ be a summarization function. For any chunk $C_i$ where $i > 1$, its fully augmented state $C'_i$ is formulated by prepending the synthesized summary of its preceding contextual window (e.g., the previous contiguous chunks or the parent hierarchical header):

$$ C'_i = S_{roll}(C_{i-n} \dots C_{i-1}) \oplus C_i $$

This augmentation guarantees that every indexed dense vector representation $\mathbf{v}_i = \text{Embed}(C'_i)$ encapsulates both the local proposition of the chunk and the global document trajectory ("velocity").

### 3.2. Phase II: Dynamic-Relevant RAG (DR-RAG) Retrieval Framework

Standard Dense Passage Retrieval (DPR) often fails on complex queries requiring cross-document reasoning. The DR-RAG module is engineered to bridge this multi-hop reasoning gap via a query-adaptive, two-stage retrieval pipeline that simulates agentic evidence-gathering without the latency overhead of continuous LLM reasoning cycles.

**3.2.1. Query Complexity Routing**
Given an initial user query $Q$, a lightweight intent classifier function $f_{route}(Q)$ evaluates the structural depth of the prompt. If the query requires factual lookup, it is routed to standard retrieval. If it necessitates multi-hop reasoning, the query is dispatched to the DR-RAG pipeline.

**3.2.2. Stage 1: Anchor Document Retrieval**
The system computes the spatial cosine similarity $\cos(\mathbf{q}, \mathbf{v}_i)$ between the embedded query vector $\mathbf{q}$ and the vector database $V$. The top-$k$ most semantically correlated chunks are retrieved to construct the set of Anchor Documents, $A = \{A_1, A_2, \dots, A_k\}$. While $A$ establishes the foundational context, it may lack the latent, weakly-linked facts required for comprehensive synthesis.

**3.2.3. Stage 2: Partial-Match Evidence Mining**
To unearth secondary supporting evidence, the system derives an expanded "Mining Query," $Q_{mine}$. This is synthesized by combining the original user query $Q$ with intermediate facts and entities extracted from the anchor set $A$:

$$ Q_{mine} = \text{Generate}(Q \oplus \text{Extract}(A)) $$

A secondary retrieval pass is executed utilizing $Q_{mine}$ to interrogate the vector database $V$, subsequently fetching a supportive set of documents $E = \{E_1, E_2, \dots, E_m\}$. The elements in $E$ capture partial semantic matches and missing correlative data that were entirely disjunct from the raw user query $Q$.

### 3.3. Synthesis and Generation Phase

The final context pool presented to the Generation LLM is the structural union of the initial anchors and the mined evidence, denoted as $Ctx = A \cup E$. The generation model conditions its probabilities on both the original query $Q$ and the enriched context $Ctx$ to formulate a highly accurate, multi-hop reasoned response $R$:

$$ P(R | Q, Ctx) = \prod_{t=1}^{T} P(Y_t | Y_{1:t-1}, Q, A, E) $$

---

## 4. Algorithmic Implementation Flow

**Algorithm 1: DR-RAG Two-Stage Retrieval Workflow**

```text
Input: User Query Q, Vector Database V, Complexity Classifier f_route
Output: Generated Response R

1.  Class ← f_route(Q)
2.  If Class == "Simple" Then
3.      A ← Retrieve_Top_K(Q, V)
4.      R ← Generate_Response(Q, A)
5.  Else
6.      // Stage 1: Anchor Retrieval
7.      A ← Retrieve_Top_K(Q, V)
8.      
9.      // Stage 2: Evidence Mining
10.     Intermediate_Facts ← Extract_Propositions(A)
11.     Q_mine ← Synthesize(Q, Intermediate_Facts)
12.     E ← Retrieve_Top_K(Q_mine, V) \ A   // Exclude already retrieved documents
13.     
14.     // Synthesis
15.     Ctx ← A ∪ E
16.     R ← Generate_Response(Q, Ctx)
17. End If
18. Return R
```

---
