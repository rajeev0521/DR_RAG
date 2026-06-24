# Research Project Methodology: Adaptive Semantic-Velocity Chunking for Efficient RAG Systems

## 1. Project Overview & Problem Analysis

Traditional Retrieval-Augmented Generation (RAG) systems suffer from rigid, fixed-size data chunking and superficial single-stage retrieval. This leads to two critical failures, especially in highly structured texts like legal contracts and academic papers:
1. **Context Fragmentation:** A chunk pulled from deep within a document loses the overarching context of earlier sections. 
2. **Multi-Hop Reasoning Gap:** Complex questions require assembling evidence from multiple disparate chunks. Often, supporting chunks lack high semantic overlap with the raw user query and remain undiscovered.

**Proposed Innovation:**
This project introduces a hybrid framework solving these issues through **Context-Augmented Agentic Chunking** and **Dynamic-Relevant RAG (DR-RAG)**. It aims to achieve high recall on complex queries typically requiring expensive Agentic frameworks, but by using efficient Small Language Models (SLMs) and two-stage retrieval.

---

## 2. System Architecture & Methodology

The methodology is divided into three core phases: Data Prep & Chunking, Intelligent Retrieval, and Synthesis & Evaluation.

### Phase 1: Context-Augmented Agentic Chunking
Instead of relying on standard character/token counts, the system dynamically parses the document based on its intrinsic structure and semantics.

1. **Document Classification Strategy:** 
   - A lightweight local classifier analyzes the document to determine its domain (e.g., Academic vs. Legal vs. Narrative).
2. **SLM-guided Boundary Detection:**
   - A fine-tuned SLM (e.g., Phi-3 or Llama-3-8B) identifies semantic boundaries. For instance, splitting legal texts by markdown headers/clauses, and academic methods by propositional logic.
3. **Running-Summary Generation (The "Velocity" Component):**
   - As chunks are created, a background process generates a concise summary of the *previous three chunks* (or the parent section's thesis).
   - This metadata summary is prepended to the current chunk before embedding. This guarantees that `Chunk N` retains the context of `Chunk N-1`, solving the isolated-clause problem.

### Phase 2: Dynamic-Relevant RAG (DR-RAG)
A two-stage retrieval pipeline that actively mines for "weak links" and supporting evidence based on initial findings.

1. **Query Complexity Routing:** 
   - A local classifier evaluates the user's query. Simple queries route straight to standard RAG. Complex queries trigger the DR-RAG multi-hop pipeline.
2. **Stage 1 (Anchor Retrieval):**
   - Standard semantic search retrieves the obvious, high-similarity "Anchor" chunks.
3. **Stage 2 (Partial-Match Mining):**
   - The system synthesizes the user query with key propositions from the Anchor chunks to construct a new expanded "Mining Query".
   - This Mining Query acts as a net to catch secondary chunks that only weakly match the original prompt but heavily contextualize the anchor documents (e.g., finding the specific academic methodology that supports an outcome found in Stage 1).

### Phase 3: Generation & Output
The retrieved primary (Anchor) and secondary (Mined) chunks, enriched with summary prefixes, are fed into the final generation LLM to construct a highly accurate, multi-hop reasoned response.

---

## 3. Technology Stack Recommendation

To ensure the project is feasible, efficient, and cost-effective, the following stack is recommended:

* **Framework:** LangChain or LlamaIndex for orchestrating the RAG pipelines.
* **Local SLMs (Chunking & Classification):** Microsoft Phi-3-Mini (3.8B) or Llama-3 (8B) quantized via Ollama to run efficiently on local hardware.
* **Embedding Model:** BGE-M3 (excellent for multi-lingual and varying chunk sizes) or Nomic Embed Text.
* **Vector Database:** Qdrant or Milvus (for fast sub-millisecond retrieval and extensive metadata filtering capabilities).
* **Generation Model:** OpenAI GPT-4o-mini, Claude 3.5 Haiku, or a local larger model like Mistral Nemo.

---

## 4. Evaluation Framework

Measuring the success of this architecture requires moving beyond standard RAG metrics.

1. **Chunking Efficacy:**
   - Evaluate against a baseline (e.g., RecursiveCharacterTextSplitter).
   - Metric: *Context Retention Score* (Does the chunk make sense in isolation?).
2. **Retrieval Performance:**
   - Datasets: HotpotQA (multi-hop reasoning benchmark) and Cuad/ContractNLI (legal contracts).
   - Metrics: `Recall@K` and `Mean Reciprocal Rank (MRR)`. The goal is a statistically significant increase in Stage-2 recall.
3. **End-to-End Generation:**
   - Utilize the **RAGAS** (RAG Assessment) framework to automatically measure:
     - *Faithfulness* (Are there hallucinations?)
     - *Answer Relevance* (Did we answer the complex prompt correctly?)
     - *Context Precision* (Did we retrieve only what we needed?)

---

## 5. Development Timeline (Approx. 5 Months)

* **Month 1: Foundation & Baselines**
  * Research literature on agentic chunking and multi-hop retrieval.
  * Set up standard RAG baseline pipeline using fixed chunks and single-stage retrieval. Get familiar with RAGAS evaluation.
* **Month 2: The Chunking Engine**
  * Develop the SLM prompt engineering and context-augmentation (summary-prefixing) loop.
  * Evaluate chunk coherence.
* **Month 3: The DR-RAG Pipeline**
  * Build the query complexity local classifier.
  * Implement the Stage 1 -> Query Expansion -> Stage 2 logic.
* **Month 4: Integration & Optimization**
  * Connect the Chunking engine to the Vector DB and integrate the DR-RAG pipeline.
  * Optimize latency (caching SLM calls, optimizing vector search).
* **Month 5: Final Evaluation & Documentation**
  * Run full comparative tests against the baseline on chosen datasets.
  * Finalize the B.Tech research report and potential publication draft.
