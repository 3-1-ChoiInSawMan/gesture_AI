from typing import Optional

from pydantic import BaseModel, Field

class SummaryRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="세션 id",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=500,
        description="요약할 최대 문서",
    )

class SummaryResponse(BaseModel):
    summary: str
    source_count: int
    session_id: Optional[str] = None