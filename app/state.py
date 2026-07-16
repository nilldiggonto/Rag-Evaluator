from dataclasses import dataclass, field


@dataclass
class DocumentRecord:
    filename: str
    token_count: int
    chunks: list[str]  # original chunk text, used for display and BM25 keyword search
    embed_texts: list[str]  # what actually gets embedded (== chunks unless contextualized)
    contextual: bool = False
    embeddings: dict[str, list[list[float]]] = field(default_factory=dict)
    embed_time_ms: dict[str, float] = field(default_factory=dict)


DOCUMENTS: dict[str, DocumentRecord] = {}
