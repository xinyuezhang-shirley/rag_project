from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ConversationCreate(BaseModel):
    datasource_id: int
    title: str = "新对话"


class ConversationResponse(BaseModel):
    id: int
    datasource_id: Optional[int]
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class QueryResult(BaseModel):
    columns: list[str] = []
    rows: list[dict] = []
    row_count: int = 0
    total_row_count: int = 0
    execution_ms: float = 0.0
    truncated: bool = False


class ChatResponse(BaseModel):
    message_id: int
    role: str = "assistant"
    content: str
    generated_sql: Optional[str] = None
    query_result: Optional[QueryResult] = None
    model: str = ""
    usage: Optional[dict] = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    generated_sql: Optional[str] = None
    execution_ms: Optional[float] = None
    row_count: Optional[int] = None


class ConversationListItem(BaseModel):
    id: int
    datasource_id: Optional[int]
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_preview: Optional[str] = None


class ConversationListResponse(BaseModel):
    items: list[ConversationListItem]
    next_cursor: Optional[str] = None