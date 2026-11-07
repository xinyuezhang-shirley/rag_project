from pydantic import BaseModel, Field
from typing import Optional


class CitationItem(BaseModel):
    index: int
    chunk_id: int
    document_id: Optional[int] = None
    source: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str
    supported: bool
    support_score: float


class CitationFragment(BaseModel):
    index: int
    chunk_id: int
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str
    supported: bool
    support_score: float


class CitationGroupItem(BaseModel):
    """作业4：同一文档下的多个引用片段聚合成一组。"""
    document_id: Optional[int] = None
    source: Optional[str] = None
    indexes: list[int] = []
    citations: list[CitationFragment] = []


class DocQARequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[int] | None = None


class DocQAResponse(BaseModel):
    message_id: int
    content: str
    answerable: bool
    citations: list[CitationItem]
    citation_groups: list[CitationGroupItem] = []
    confidence: float
    confidence_level: str
    degraded: list[str] = []
    query_used: str = ""
    query_rewritten: bool = False
    model: str = ""
    usage: Optional[dict] = None