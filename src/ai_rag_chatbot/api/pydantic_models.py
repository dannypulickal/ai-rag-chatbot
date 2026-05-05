"""Pydantic models for API request and response validation.

Defines data structures for:
- API input: user queries and model selection
- API output: AI responses with session tracking
- Document metadata: file information
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    """Supported OpenAI models for chat interactions.

    Attributes:
        GPT4_O: Full GPT-4 Omni model (more capable, higher cost)
        GPT4_O_MINI: Lightweight GPT-4 Omni Mini (faster, lower cost)
    """
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        question (str): The user's query or prompt.
        session_id (str, optional): Unique session identifier for conversation continuity.
                                   If None, a new session ID is generated.
        model (ModelName): The AI model to use for the response
                          (default: GPT4_O_MINI).
    """
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        answer (str): The AI-generated response to the user's question.
        session_id (str): The session identifier for tracking conversation continuity.
        model (ModelName): The model that generated the response.
    """
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    """Response model for document metadata.

    Attributes:
        id (int): Unique database identifier for the document.
        filename (str): Original filename of the uploaded document.
        upload_timestamp (datetime): Timestamp when the document was uploaded.
    """
    id: int
    filename: str
    upload_timestamp: datetime

# class DeleteFileRequest(BaseModel):
#     file_id: int
