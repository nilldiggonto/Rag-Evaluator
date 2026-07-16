# RAG Evaluator

A local web tool for sanity-checking a RAG setup *before* building a full pipeline around it.
Upload a text/PDF file **or paste text**, ask a few questions (optionally with a known
"expected keyword" so it can be scored), pick which locally-installed Ollama embedding models
and which retrieval strategies (Top-K, MMR, Hybrid, HyDE) to try — optionally with **Contextual
Retrieval** at indexing time — and get a side-by-side metrics comparison. The app checks whether
Ollama is installed and running on startup and tells you on the page if it isn't.

It's built to **teach**: a built-in FAQ explains every concept (tokens, embeddings, chunking,
each strategy and where it breaks), the loading states narrate what's happening under the hood,
and a slide-out **Ollama panel** lists your models and explains each one's metadata (context
length, embedding length, parameter size, quantization) in plain language. A **light/dark theme
toggle** sits in the top bar.

This won't give you a perfectly accurate answer — it's a rough, local, no-cloud way to get
a first read on "is my chunk size reasonable" and "which embedding model / algorithm looks
more promising for this document" before committing to a real implementation.
<img width="1875" height="851" alt="image" src="https://github.com/user-attachments/assets/1a55962a-81ad-484c-8e94-aa2437d966dc" />
<img width="1057" height="637" alt="image" src="https://github.com/user-attachments/assets/53009233-914a-449e-9733-43b4645d95a7" />
<img width="1048" height="666" alt="image" src="https://github.com/user-attachments/assets/193e2376-6cd3-4e95-bc61-e779fdc1a465" />
<img width="836" height="763" alt="image" src="https://github.com/user-attachments/assets/591df762-6f22-4907-80ea-82e8aaa3afd1" />
<img width="1878" height="822" alt="image" src="https://github.com/user-attachments/assets/d9184ea3-a49c-4c32-afab-828ec697716c" />

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with at least:
  - one **embedding** model pulled (e.g. `ollama pull nomic-embed-text`)
  - one **generation/chat** model pulled (e.g. `ollama pull llama3.2`)

## Setup

```bash
pip install -r requirements.txt
```


## Run


```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** in a browser.

## How to use

1. **Upload & process** — pick a `.txt` or `.pdf` file, set a chunk size / overlap (in tokens),
   check which embedding models to test, click "Process Document". This chunks the document
   and embeds every chunk with every selected model — the first run per model can be slow
   while Ollama loads that model into memory.
2. **Questions & algorithms** — add one or more questions. For any question where you already
   know part of the answer, fill in "Expected keyword" (a distinctive word/phrase you know
   only appears in the chunk that should answer it) to unlock accuracy scoring for that
   question. Pick one or more algorithms (see below), then "Run Evaluation". You'll get a
   results table — one row per (embedding model, algorithm) combination — sorted by
   relevance. Click a row to expand per-question detail and see the actual retrieved chunk
   text.
3. **Generate a grounded answer** — pick one result combination, one question, and a
   generation model, then "Generate" to see an actual LLM answer built only from the
   retrieved context (not run for every combination automatically, since that would be slow).

## What the algorithms mean

- **Top-K** — the baseline. Grab the k chunks whose embedding is closest to the question's embedding.
- **MMR (Maximal Marginal Relevance)** — like Top-K, but each pick after the first is penalized
  for being too similar to chunks already picked, trading some relevance for less redundancy.
  Implemented by hand in `services/retrieval.py`, same algorithm worked out during the learning session.
- **Hybrid** — merges Top-K's meaning-based ranking with a hand-written **BM25** keyword-search
  ranking (classic term-frequency/inverse-document-frequency scoring, no embeddings involved),
  combined via **Reciprocal Rank Fusion**. Helps when the answer hinges on an exact word/name
  the embedding model didn't weight heavily.
- **HyDE (Hypothetical Document Embeddings)** — asks a chat model to write a short hypothetical
  answer to the question first, embeds *that* instead of the raw question, then runs Top-K.
  Needs a generation model selected, since it makes one extra LLM call per question before retrieval.

**Contextual Retrieval** (a processing-time option, not a query-time strategy) — before embedding,
a chat model writes a 1–2 sentence blurb describing what each chunk is about and prepends it, so
context-poor chunks embed far better. This is Anthropic's "Contextual Retrieval" idea. It makes one
LLM call *per chunk*, so indexing is slow; the document text is truncated to ~2000 tokens for each
context prompt to avoid overflowing small chat-model context windows.

Deliberately **not** included: cross-encoder/LLM-based re-ranking. It's a real technique (see
Roadmap below), but reliably parsing a ranked order out of a small local model's free-text
output turned out to be too flaky to ship as a trustworthy metric.

## What the metrics mean

- **Avg Relevance** — mean cosine similarity between the question and each retrieved chunk.
  Higher means the retrieved chunks are more semantically related to the question.
- **Avg Redundancy** — mean pairwise cosine similarity *among* the retrieved chunks
  themselves. Higher means the retrieved chunks are near-duplicates of each other (wasted
  retrieval budget); lower means more varied coverage. MMR is designed to lower this, but
  isn't guaranteed to — that's exactly why this tool measures it instead of assuming it.
- **Hit Rate** — for questions with an "Expected keyword" set, the fraction where that
  keyword actually showed up in the retrieved chunks. This is the only metric here that's a
  real accuracy check rather than a heuristic; it needs you to supply ground truth.
- **Avg Latency (ms)** — how long retrieval took, per question, for that combination.

## Known limitations (v1)

- **In-memory only.** Everything resets when the server restarts — there's no database.
  This is meant for quick local evaluation runs, not as a persistent knowledge base.
- **Brute-force similarity search.** Fine for a single uploaded document's worth of chunks;
  not meant for huge corpora (a real vector database with an ANN index would be needed there).
- **Tokenizer is an approximation.** Chunking uses `tiktoken`'s `cl100k_base` encoding for
  all models, not each model's exact tokenizer, so token counts are estimates.
- **Generation runs one combination at a time**, not across the whole comparison matrix,
  to keep things fast.

## Roadmap ideas

- Persistent storage (so processed documents survive a restart).
- Cross-encoder or LLM-based re-ranking as a second, more precise pass over top candidates
  (needs a more reliable way to extract a ranking than asking a small local model to print one).
- Semantic/sentence-aware chunking as an alternative to fixed-size chunking.
- CSV import for larger question sets.
- Light/dark theme toggle (currently follows the OS preference only).
