from .retrieval import cosine_similarity


def avg_relevance(query_embedding, retrieved_embeddings: list[list[float]]) -> float:
    if not retrieved_embeddings:
        return 0.0
    return sum(cosine_similarity(query_embedding, emb) for emb in retrieved_embeddings) / len(
        retrieved_embeddings
    )


def avg_redundancy(retrieved_embeddings: list[list[float]]) -> float:
    n = len(retrieved_embeddings)
    if n < 2:
        return 0.0
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(cosine_similarity(retrieved_embeddings[i], retrieved_embeddings[j]))
    return sum(pairs) / len(pairs)


def parse_keywords(expected_keyword: str | None) -> list[str]:
    """Split a comma-separated 'expected keyword' field into individual keywords,
    so 'Django, FastAPI' becomes two acceptable keywords rather than one literal
    phrase that includes the comma."""
    if not expected_keyword:
        return []
    return [k.strip().lower() for k in expected_keyword.split(",") if k.strip()]


def check_hit(expected_keyword: str | None, retrieved_texts: list[str]) -> dict | None:
    """Returns None when no keyword was given, otherwise a breakdown of which of the
    comma-separated keywords actually appeared in the retrieved chunks. Counts as a
    hit if ANY keyword matched (finding one confirms the relevant chunk was retrieved)."""
    keywords = parse_keywords(expected_keyword)
    if not keywords:
        return None
    haystack = "\n".join(t.lower() for t in retrieved_texts)
    matched = [k for k in keywords if k in haystack]
    return {"hit": len(matched) > 0, "keywords": keywords, "matched": matched}
