from pydantic import BaseModel


class QuestionItem(BaseModel):
    question: str
    expected_keyword: str | None = None


class EvaluateRequest(BaseModel):
    document_id: str
    questions: list[QuestionItem]
    algorithms: list[str]
    embedding_models: list[str]
    generation_model: str | None = None  # required if "hyde" is among algorithms
    k: int = 3
    fetch_k: int = 8
    lambda_mult: float = 0.5


class GenerateRequest(BaseModel):
    document_id: str
    question: str
    embedding_model: str
    algorithm: str
    generation_model: str
    k: int = 3
    fetch_k: int = 8
    lambda_mult: float = 0.5
