import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    token_ids = _encoding.encode(text)

    chunks = []
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start : start + chunk_size]
        chunks.append(_encoding.decode(chunk_ids))
        if start + chunk_size >= len(token_ids):
            break

    return chunks


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))
