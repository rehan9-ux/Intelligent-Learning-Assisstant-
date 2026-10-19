# Intelligent Learning Assistant (ILA)
## Comprehensive Study Guide & Viva / Professor Presentation Reference

---

### Project Overview
* **Project Name**: Intelligent Learning Assistant (ILA)
* **Authors**: Krish Shandilya (24UCS156) & Rehan Khan (24UCS277)
* **Core Value Proposition**: A 100% offline, privacy-first, adaptive AI tutoring system combining Local RAG, Open-Weight LLMs, and Cognitive Machine Learning algorithms (Bayesian Knowledge Tracing & SM-2 Spaced Repetition).

---

## Pillar 1: Retrieval-Augmented Generation (RAG) & Vector Math

### 1.1 What is RAG and Why Not Fine-Tuning?
* **Fine-Tuning**: Modifies the internal weights of an LLM. It is expensive, prone to catastrophic forgetting, causes hallucinations, and takes hours/days every time new lecture notes are added.
* **RAG (Retrieval-Augmented Generation)**: Keeps the model weights static. When a student asks a question, the system searches local vector storage for the top-$k$ most relevant text passages from their uploaded notes and injects them directly into the LLM's prompt as grounded reference material.
* **Result**: Zero hallucinations, instantaneous addition of new textbooks without training, and page-level source citations.

### 1.2 Chunking & Text Splitting
* **Why Chunking?**: LLMs have finite context limits, and embedding an entire 500-page textbook as a single vector dilutes semantic specificity.
* **Our Parameters**:
  * **Chunk Size**: `512` tokens (optimal semantic granularity).
  * **Chunk Overlap**: `64` tokens (ensures definitions, formulas, or theorems spanning chunk boundaries are preserved in both chunks).
* **Metadata Tagging**: Every chunk in ChromaDB is indexed with `{source_file, page_number, chunk_index, doc_hash}` to enable transparent citation display.

### 1.3 High-Dimensional Vector Embeddings
* **Model**: `nomic-embed-text` (137M parameters, 274 MB).
* **Vector Space**: Transforms arbitrary English text into a dense vector of **768 floating-point numbers**.
* **Semantic Geometry**: Sentences with similar conceptual meaning (e.g., *"Newton's 1st Law"* and *"Principle of Inertia"*) map to vectors pointing in almost the same direction in 768-dimensional space.

### 1.4 Similarity Mathematics: Cosine Similarity
To measure semantic similarity between a student's query vector $\mathbf{A}$ and a document chunk vector $\mathbf{B}$:

$$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* **Range**: $0.0$ (completely orthogonal/unrelated) to $1.0$ (identical meaning).
* **Distance Metric**: In ChromaDB, Cosine Distance is defined as:

$$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$

* Chunks with distance $< 0.35$ are highly relevant. Distances $> 1.5$ indicate out-of-domain queries.

### 1.5 Vector Database (ChromaDB) & HNSW
* **ChromaDB**: An open-source, embedded, serverless vector store running directly in Python.
* **HNSW (Hierarchical Navigable Small World)**: The indexing algorithm used by ChromaDB. Instead of scanning every vector $O(N)$, HNSW builds a multi-layered graph that allows finding nearest neighbors in $O(\log N)$ logarithmic time.

---

## Pillar 2: Local LLMs & Inference Architecture

### 2.1 Model Quantization (GGUF Format)
* Standard deep learning weights are stored in 16-bit floating point (FP16), requiring ~16 GB of memory for an 8B model.
* **4-bit Quantization (`Q4_K_M`)**: Compresses weights from 16-bit floats to 4-bit integers.
* **Benefits**: Cuts RAM/VRAM consumption by **~70%** (from 16 GB down to 5.2 GB) with less than a 1% drop in academic benchmark accuracy.

### 2.2 GPU Acceleration vs. CPU Offloading
* On an **NVIDIA RTX 4050 (6 GB VRAM)**:
  * `qwen3:8b` runs with **~70% GPU / 30% CPU** layer split (~5.3 seconds response).
  * `phi4-mini` (3.8B) runs **100% on GPU VRAM** (~1.5 seconds response).
* On a **Non-GPU / CPU-Only machine**:
  * Runs via AVX2 CPU instructions at ~3–8 tokens/second.

### 2.3 Real-Time Streaming vs. Non-Streaming
* **Content Equivalence**: The output text in streaming mode is **100% mathematically identical** to non-streaming mode.
* **The Mechanism**:
  * **Non-Streaming**: Blocks the application thread for 8–12 seconds until all tokens are generated, then dumps the full block of text.
  * **Streaming**: Uses **NDJSON (Newline Delimited JSON)** over HTTP. As each token is generated (~25ms), it is immediately emitted over a socket and rendered on screen via Python generators (`yield`).
* **Metric**: Cuts **Time to First Token (TTFT)** from ~10,000ms down to **~350ms**.

---

## Pillar 3: Cognitive ML & Adaptive Learning Algorithms

*This is the major academic component that elevates ILA beyond simple chatbots.*

### 3.1 Bayesian Knowledge Tracing (BKT)
BKT is a Hidden Markov Model (HMM) used to model student knowledge state over time.

```
                  ┌────────────────────────┐
                  │   Hidden State:        │
                  │   Mastered vs Unlearned│
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │   Observable Action:   │
                  │   Correct vs Incorrect │
                  └────────────────────────┘
```

#### The 4 Core Parameters:
1. **$P(L_0)$ (Initial Mastery)**: The prior probability that a student already understands the topic before answering questions.
2. **$P(T)$ (Transition / Learning Rate)**: The probability that a student transitions from the unlearned state to the learned state after a practice attempt.
3. **$P(G)$ (Guess Rate)**: The probability that an unlearned student guesses the correct answer by luck (typically 0.25 on a 4-option MCQ).
4. **$P(S)$ (Slip Rate)**: The probability that a student who actually knows the concept makes a careless mistake.

#### Updating Student Mastery on Observation:
When a student answers a question at step $t$:

* **If Answer is Correct ($obs = 1$)**:
$$P(L_t \mid \text{correct}) = \frac{P(L_{t-1}) \cdot (1 - P(S))}{P(L_{t-1}) \cdot (1 - P(S)) + (1 - P(L_{t-1})) \cdot P(G)}$$

* **If Answer is Incorrect ($obs = 0$)**:
$$P(L_t \mid \text{incorrect}) = \frac{P(L_{t-1}) \cdot P(S)}{P(L_{t-1}) \cdot P(S) + (1 - P(L_{t-1})) \cdot (1 - P(G))}$$

* **Accounting for Learning Transition**:
$$P(L_{t+1}) = P(L_t \mid obs) + (1 - P(L_t \mid obs)) \cdot P(T)$$

* When $P(L_t) \ge 0.75$, the student is considered to have **Mastered** the topic.

---

### 3.2 Spaced Repetition: The SM-2 Algorithm
Based on the **Ebbinghaus Forgetting Curve**, which demonstrates that human memory decays exponentially unless reviewed at increasing intervals.

```
Memory Retention
  100% │  \
       │    \   (Review 1)
       │      \───\
       │            \   (Review 2)
       │              \─────\
   0%  └────────────────────────► Time (Days)
```

#### The SM-2 Mathematical Formulation:
For each topic, the system tracks:
* **$I(n)$**: Interval in days until next review.
* **$EF$**: Ease Factor (starts at $2.5$, minimum $1.3$).
* **$q$**: Performance quality grade ($0$ = blackout, $5$ = perfect recall).

$$\begin{aligned}
I(1) &= 1 \text{ day} \\
I(2) &= 6 \text{ days} \\
I(n) &= I(n-1) \times EF \quad (\text{for } n > 2)
\end{aligned}$$

#### Updating the Ease Factor:
$$EF' = \max\left(1.3, \; EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))\right)$$

If the student fails ($q < 3$), the repetition counter resets to $n = 1$, scheduling immediate review.

---

## Pillar 4: Software Architecture & Data Engineering

### 4.1 System Block Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          STUDENT FRONTEND                              │
│         Streamlit Web UI (Chat, PDF Upload, Quiz, Analytics)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                           CORE ILA ENGINE                              │
│                                                                        │
│  ┌───────────────────────┐                  ┌───────────────────────┐  │
│  │   Document Ingestion  │                  │   RAG Query Engine    │  │
│  │   (PyMuPDF + Chunks)  │                  │   (Grounded Prompts)  │  │
│  └───────────┬───────────┘                  └───────────▲───────────┘  │
│              │                                          │              │
│  ┌───────────▼───────────┐                  ┌───────────┴───────────┐  │
│  │  ChromaDB (Vectors)   │◄─────────────────┤  Ollama LLM Runtime   │  │
│  │  (nomic-embed-text)   │                  │  (qwen3:8b / 4-bit)   │  │
│  └───────────────────────┘                  └───────────▲───────────┘  │
│                                                         │              │
│  ┌──────────────────────────────────────────────────────┴───────────┐  │
│  │      Adaptive Engine (Bayesian Knowledge Tracing + SM-2)         │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼─────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────┐
│                      SQLite PERSISTENCE LAYER                          │
│   • student_profiles   • quiz_results   • spaced_repetition   • docs   │
│   • Indexed on: CREATE INDEX idx_student_topic                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 SQLite Optimization: Composite B-Tree Indexing
* **Table**: `quiz_results`
* **Index**: `CREATE INDEX idx_student_topic ON quiz_results(student_id, topic)`
* **Rationale**: Recomputing Bayesian mastery curves requires filtering hundreds of past responses by topic. The composite B-Tree index reduces query complexity from $O(N)$ full table scan to $O(\log N)$ logarithmic index lookup.

### 4.3 SHA-256 Deduplication Caching
* Before chunking and embedding, `compute_file_hash()` calculates the SHA-256 fingerprint of the uploaded file bytes.
* If the hash matches an entry in SQLite `indexed_documents`, the system bypasses the entire embedding pipeline, preventing duplicate GPU computation.

### 4.4 Multi-Device LAN Deployment (Host-Client Model)
* The host laptop (with GPU) binds Streamlit/FastAPI to all network interfaces (`0.0.0.0:8501`).
* Other devices (phones, budget laptops) on the same local Wi-Fi / hotspot connect via `http://<HOST_IP>:8501`.
* **Zero GPU or Python setup is required on client devices.**

---

## Pillar 5: Top 10 B.Tech Viva Questions & Model Answers

### Q1: What makes your project different from ChatGPT or Gemini?
> **Answer**: *"ChatGPT is a cloud-based generalist model that requires paid API subscriptions, sends private student data to external servers, and frequently hallucinates formulas not in the syllabus. ILA runs 100% locally and offline, answers strictly from course PDFs, cites exact page numbers, and uses classical Bayesian Knowledge Tracing to mathematically track and adapt to student weakness areas."*

### Q2: Why did you choose ChromaDB over relational databases for document retrieval?
> **Answer**: *"Relational databases perform exact keyword matching (e.g. `LIKE '%inertia%'`), which fails when queries use synonyms or natural language. ChromaDB uses high-dimensional vector embeddings and HNSW graphs to perform semantic search based on mathematical concept similarity regardless of the exact wording."*

### Q3: How do you prevent the LLM from hallucinating answers?
> **Answer**: *"Through a three-layer defense: 1) Distance threshold filtering: queries with cosine distance $> 1.5$ are rejected before prompting. 2) Strict system prompt constraints: the LLM is explicitly penalized for using external knowledge. 3) Mandatory citation formatting: the LLM must anchor its explanation to retrieved passage tags."*

### Q4: Explain the role of the 64-token chunk overlap.
> **Answer**: *"If a paragraph or equation is split right at the 512-token limit, the second half loses its prerequisite context. Overlap ensures boundary sentences exist in both neighboring chunks, guaranteeing no loss of semantic meaning during retrieval."*

### Q5: What is the computational advantage of running `nomic-embed-text` separately from `qwen3:8b`?
> **Answer**: *"Embedding models are lightweight encoder-only networks (~274 MB) that convert text to vectors in milliseconds. LLMs are heavy autoregressive decoders (~5.2 GB). Running them as decoupled services allows us to vectorize documents rapidly on CPU/GPU without keeping the large generative model loaded."*

### Q6: How does Bayesian Knowledge Tracing handle a student guessing an MCQ?
> **Answer**: *"BKT includes a dedicated Guess Parameter $P(G)$. When updating mastery after a correct answer, the formula factors in the possibility that the correct answer was a random guess (25% on 4-option MCQs), preventing a single lucky answer from prematurely setting mastery to 100%."*

### Q7: What happens when a student asks a question about a completely unrelated topic?
> **Answer**: *"Our semantic distance filter calculates the vector distance between the query and all stored document chunks. If no chunk falls within the similarity threshold, ILA bypasses generative synthesis and safely informs the student: 'Based on your uploaded material, I don't have enough information to answer this topic.'"*

### Q8: Why did you use SQLite instead of MongoDB or PostgreSQL?
> **Answer**: *"For an offline, self-contained desktop system, SQLite provides zero-configuration, serverless, single-file ACID storage with zero background daemon overhead. It comes natively with Python and handles indexed lookups in microseconds."*

### Q9: What is Time to First Token (TTFT) and why is streaming crucial?
> **Answer**: *"TTFT is the latency between submitting a prompt and rendering the first word. Non-streaming has a TTFT of ~10 seconds because it waits for the full answer. Streaming achieves a TTFT of ~350ms by rendering NDJSON tokens in real-time as the GPU computes them."*

### Q10: If you have to deploy this in a college with poor infrastructure, how does it scale?
> **Answer**: *"Through our Local LAN Host model. One mid-range laptop or lab workstation hosts the backend, while 10+ students on basic mobile phones or old PCs access the tutor simultaneously over local Wi-Fi without needing GPUs or internet access."*

---

*Compiled for B.Tech Mini Project Evaluation · Krish Shandilya & Rehan Khan*
