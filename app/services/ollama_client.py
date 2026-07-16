import shutil

import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def list_models() -> dict:
    if not is_ollama_running():
        return {
            "ollama_running": False,
            "ollama_installed": is_ollama_installed(),
            "embedding_models": [],
            "generation_models": [],
        }

    tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags").json()["models"]

    embedding_models = []
    generation_models = []

    for model in tags:
        name = model["name"]
        info = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"name": name}).json()
        capabilities = info.get("capabilities", [])
        if "embedding" in capabilities:
            embedding_models.append(name)
        if "completion" in capabilities:
            generation_models.append(name)

    return {
        "ollama_running": True,
        "ollama_installed": True,
        "embedding_models": embedding_models,
        "generation_models": generation_models,
    }


def model_info(name: str) -> dict:
    """Reads a single model's metadata via /api/show and pulls out the fields
    we explained during the learning sessions (context length, embedding length,
    parameter size, quantization, capabilities)."""
    raw = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"name": name}).json()
    details = raw.get("details", {})
    info = raw.get("model_info", {})

    context_length = None
    embedding_length = None
    for key, value in info.items():
        if key.endswith(".context_length"):
            context_length = value
        elif key.endswith(".embedding_length"):
            embedding_length = value

    return {
        "name": name,
        "capabilities": raw.get("capabilities", []),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "context_length": context_length,
        "embedding_length": embedding_length,
    }


def embed(text: str, model: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
    )
    return response.json()["embedding"]


def generate(prompt: str, model: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    return response.json()["response"]


def hyde_prompt(question: str) -> str:
    return f"""Write a short, plausible passage that could answer the following question.
Do not say you don't know - just write a hypothetical answer in a few sentences,
even if you are not sure it is correct.

Question: {question}

Hypothetical answer:"""


def contextual_prompt(document_excerpt: str, chunk: str) -> str:
    return f"""<document>
{document_excerpt}
</document>

Here is a chunk taken from that document:
<chunk>
{chunk}
</chunk>

Give a short (1-2 sentence) context that situates this chunk within the overall
document, so the chunk is easier to find in a search. Answer with the context only,
no preamble.

Context:"""
