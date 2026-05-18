from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    job_status: str
    steps: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class GenericMessageResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StubDataResponse(BaseModel):
    job_id: str
    status: str
    data: Dict[str, Any]
