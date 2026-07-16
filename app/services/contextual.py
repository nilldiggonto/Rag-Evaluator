from .ollama_client import contextual_prompt, generate

# Cap how much of the document we stuff into each contextualization prompt, so a
# large document does not overflow the chat model's context window. This is a rough
# guard - a production system would summarize the document instead of truncating.
MAX_DOCUMENT_TOKENS = 2000


def _truncate_document(text: str) -> str:
    from tiktoken import get_encoding

    enc = get_encoding("cl100k_base")
    ids = enc.encode(text)
    if len(ids) <= MAX_DOCUMENT_TOKENS:
        return text
    return enc.decode(ids[:MAX_DOCUMENT_TOKENS])


def contextualize_chunks(full_text: str, chunks: list[str], generation_model: str) -> list[str]:
    """For each chunk, asks the chat model to write a 1-2 sentence context blurb
    situating it in the wider document, and prepends that to the chunk before it
    gets embedded. This is Anthropic's 'Contextual Retrieval' idea: a chunk like
    'It dropped 15% that quarter' embeds much better once it carries 'This is about
    ACME's Q3 2023 revenue' with it. Costs one LLM call per chunk, so it is slow."""
    document_excerpt = _truncate_document(full_text)
    contextualized = []
    for chunk in chunks:
        context = generate(contextual_prompt(document_excerpt, chunk), generation_model).strip()
        contextualized.append(f"{context}\n\n{chunk}")
    return contextualized
