import math
import re
from collections import Counter


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def retrieve_topk(query_embedding, chunk_embeddings: list[list[float]], k: int) -> list[int]:
    scored = [
        (i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(chunk_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]]


def retrieve_mmr(
    query_embedding,
    chunk_embeddings: list[list[float]],
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> list[int]:
    candidate_indices = retrieve_topk(query_embedding, chunk_embeddings, fetch_k)
    remaining = [(i, chunk_embeddings[i]) for i in candidate_indices]

    selected_indices: list[int] = []
    selected_embeds: list[list[float]] = []

    while remaining and len(selected_indices) < k:
        best_score = None
        best_item = None
        for i, emb in remaining:
            relevance = cosine_similarity(query_embedding, emb)
            redundancy = max((cosine_similarity(emb, s) for s in selected_embeds), default=0.0)
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_item = (i, emb)

        selected_indices.append(best_item[0])
        selected_embeds.append(best_item[1])
        remaining = [(i, emb) for i, emb in remaining if i != best_item[0]]

    return selected_indices


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Classic keyword-search scoring (no embeddings involved) - rewards chunks
    that actually contain the query's words, weighted by how rare/informative
    each word is across the document and normalized for chunk length."""
    tokenized_docs = [_tokenize(doc) for doc in documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    n_docs = len(documents)

    doc_freq = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freq[term] += 1

    query_terms = _tokenize(query)
    scores = []
    for doc, length in zip(tokenized_docs, doc_lengths):
        term_counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            denom = tf + k1 * (1 - b + b * length / avg_len) if avg_len else tf
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return scores


def retrieve_hybrid(
    query_text: str,
    query_embedding: list[float],
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    k: int,
    fetch_k: int,
    rrf_k: int = 60,
) -> list[int]:
    """Combines vector similarity (meaning-based) and BM25 (keyword-based) rankings
    using Reciprocal Rank Fusion: each chunk's fused score is the sum, across both
    rankings it appears in, of 1/(rrf_k + rank). This catches cases pure vector
    search misses, e.g. an exact name/code the embedding model doesn't weight highly."""
    vector_ranked = retrieve_topk(query_embedding, chunk_embeddings, fetch_k)

    bm25 = bm25_scores(query_text, chunk_texts)
    bm25_ranked = sorted(range(len(bm25)), key=lambda i: bm25[i], reverse=True)[:fetch_k]

    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(vector_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (rrf_k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in merged[:k]]
