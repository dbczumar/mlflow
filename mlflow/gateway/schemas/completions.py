from typing import Dict, List, Optional

from pydantic import Field

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import RouteType
from mlflow.gateway.schemas.chat import BaseRequestPayload


class BaseRequestPayload(RequestModel):
    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[List[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)


class RequestPayload(BaseRequestPayload):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "hello",
                "temperature": 0.0,
                "max_tokens": 64,
                "stop": ["END"],
                "candidate_count": 1,
            }
        }


class CandidateMetadata(ResponseModel):
    finish_reason: Optional[str] = None


class Candidate(ResponseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None


class Metadata(ResponseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str
    route_type: RouteType


class ResponsePayload(ResponseModel):
    candidates: List[Candidate]
    metadata: Metadata

    class Config:
        schema_extra = {
            "example": {
                "candidates": [
                    {
                        "text": "hello world",
                        "metadata": {
                            "finish_reason": "stop",
                        },
                    }
                ],
                "metadata": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                    "model": "gpt-3.5-turbo",
                    "route_type": "llm/v1/completions",
                },
            }
        }
