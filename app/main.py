import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import state
from .schemas import EvaluateRequest, GenerateRequest
from .services import chunking, contextual, documents, metrics, ollama_client, retrieval

app = FastAPI(title="RAG Evaluator")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/learn", response_class=HTMLResponse)
def learn(request: Request):
    return templates.TemplateResponse(request, "learn.html")


@app.get("/api/models")
def get_models():
    return ollama_client.list_models()


@app.get("/api/model-info")
def get_model_info(name: str):
    if not ollama_client.is_ollama_running():
        raise HTTPException(503, "Ollama is not running")
    return ollama_client.model_info(name)


@app.post("/api/documents")
async def create_document(
    embedding_models: list[str] = Form(...),
    chunk_size: int = Form(300),
    overlap: int = Form(50),
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    filename: str = Form("pasted-text.txt"),
    contextual_enabled: bool = Form(False),
    context_model: str | None = Form(None),
):
    if overlap >= chunk_size:
        raise HTTPException(400, "overlap must be smaller than chunk_size")

    if file is not None and file.filename:
        content = await file.read()
        source_name = file.filename
        raw_text = documents.extract_text(source_name, content)
    elif text and text.strip():
        source_name = filename
        raw_text = text
    else:
        raise HTTPException(400, "provide either a file or pasted text")

    if not raw_text.strip():
        raise HTTPException(400, "the document appears to be empty (no extractable text)")

    chunks = chunking.chunk_text(raw_text, chunk_size, overlap)

    if contextual_enabled:
        if not context_model:
            raise HTTPException(400, "a generation model is required for contextual retrieval")
        embed_texts = contextual.contextualize_chunks(raw_text, chunks, context_model)
    else:
        embed_texts = chunks

    document_id = str(uuid.uuid4())
    record = state.DocumentRecord(
        filename=source_name,
        token_count=chunking.count_tokens(raw_text),
        chunks=chunks,
        embed_texts=embed_texts,
        contextual=contextual_enabled,
    )

    per_model = {}
    for model in embedding_models:
        start = time.perf_counter()
        embeddings = [ollama_client.embed(t, model) for t in embed_texts]
        elapsed_ms = (time.perf_counter() - start) * 1000

        record.embeddings[model] = embeddings
        record.embed_time_ms[model] = elapsed_ms
        per_model[model] = {
            "dimension": len(embeddings[0]) if embeddings else 0,
            "embed_time_ms": round(elapsed_ms, 1),
        }

    state.DOCUMENTS[document_id] = record

    return {
        "document_id": document_id,
        "filename": source_name,
        "token_count": record.token_count,
        "chunk_count": len(chunks),
        "contextual": contextual_enabled,
        "per_model": per_model,
    }


def _get_document(document_id: str) -> state.DocumentRecord:
    record = state.DOCUMENTS.get(document_id)
    if record is None:
        raise HTTPException(404, "document not found (server may have restarted)")
    return record


ALGORITHMS_NEEDING_GENERATION_MODEL = {"hyde"}


def _run_retrieval(record, model, algorithm, question, k, fetch_k, lambda_mult, generation_model):
    """Returns (retrieved_indices, question_embedding). question_embedding is always
    the embedding of the real question (used for the relevance metric), even for
    algorithms like HyDE that search using a different embedding."""
    question_embedding = ollama_client.embed(question, model)
    chunk_embeddings = record.embeddings[model]

    if algorithm == "topk":
        indices = retrieval.retrieve_topk(question_embedding, chunk_embeddings, k)
    elif algorithm == "mmr":
        indices = retrieval.retrieve_mmr(question_embedding, chunk_embeddings, k, fetch_k, lambda_mult)
    elif algorithm == "hybrid":
        indices = retrieval.retrieve_hybrid(
            question, question_embedding, record.chunks, chunk_embeddings, k, fetch_k
        )
    elif algorithm == "hyde":
        if not generation_model:
            raise HTTPException(400, "generation_model is required when using the HyDE algorithm")
        hypothetical = ollama_client.generate(ollama_client.hyde_prompt(question), generation_model)
        hyde_embedding = ollama_client.embed(hypothetical, model)
        indices = retrieval.retrieve_topk(hyde_embedding, chunk_embeddings, k)
    else:
        raise HTTPException(400, f"unknown algorithm: {algorithm}")

    return indices, question_embedding


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    record = _get_document(req.document_id)

    for model in req.embedding_models:
        if model not in record.embeddings:
            raise HTTPException(400, f"document was not embedded with model: {model}")

    if not req.generation_model and ALGORITHMS_NEEDING_GENERATION_MODEL & set(req.algorithms):
        raise HTTPException(400, "generation_model is required when using the HyDE algorithm")

    results = []
    for model in req.embedding_models:
        for algorithm in req.algorithms:
            per_question = []
            for item in req.questions:
                start = time.perf_counter()
                indices, question_embedding = _run_retrieval(
                    record, model, algorithm, item.question, req.k, req.fetch_k, req.lambda_mult, req.generation_model
                )
                latency_ms = (time.perf_counter() - start) * 1000

                retrieved_texts = [record.chunks[i] for i in indices]
                retrieved_embeddings = [record.embeddings[model][i] for i in indices]

                kw = metrics.check_hit(item.expected_keyword, retrieved_texts)
                per_question.append(
                    {
                        "question": item.question,
                        "relevance": round(metrics.avg_relevance(question_embedding, retrieved_embeddings), 4),
                        "redundancy": round(metrics.avg_redundancy(retrieved_embeddings), 4),
                        "hit": kw["hit"] if kw else None,
                        "hit_keywords": kw["keywords"] if kw else [],
                        "hit_matched": kw["matched"] if kw else [],
                        "latency_ms": round(latency_ms, 1),
                        "retrieved_chunks": retrieved_texts,
                    }
                )

            hits = [q["hit"] for q in per_question if q["hit"] is not None]
            results.append(
                {
                    "embedding_model": model,
                    "algorithm": algorithm,
                    "avg_relevance": round(sum(q["relevance"] for q in per_question) / len(per_question), 4),
                    "avg_redundancy": round(sum(q["redundancy"] for q in per_question) / len(per_question), 4),
                    "hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
                    "avg_latency_ms": round(sum(q["latency_ms"] for q in per_question) / len(per_question), 1),
                    "per_question": per_question,
                }
            )

    return {"results": results}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    record = _get_document(req.document_id)
    if req.embedding_model not in record.embeddings:
        raise HTTPException(400, f"document was not embedded with model: {req.embedding_model}")

    start = time.perf_counter()
    indices, _ = _run_retrieval(
        record, req.embedding_model, req.algorithm, req.question, req.k, req.fetch_k, req.lambda_mult, req.generation_model
    )
    retrieved_chunks = [record.chunks[i] for i in indices]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know based on the given context."

Context:
{context}

Question: {req.question}

Answer:"""

    answer = ollama_client.generate(prompt, req.generation_model)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
    }
