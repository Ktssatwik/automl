from typing import Any, Dict

from pydantic import BaseModel, Field


class RunAutoMLRequest(BaseModel):
    job_id: str = Field(..., description="Job ID returned by /upload-csv")


class PredictRequest(BaseModel):
    record: Dict[str, Any] = Field(..., description="Single record for prediction")
