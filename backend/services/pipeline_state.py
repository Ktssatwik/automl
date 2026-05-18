import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List

PIPELINE_STEPS: List[str] = [
    "csv_understanding",
    "problem_type_detection",
    "domain_understanding",
    "eda",
    "preprocessing",
    "train_test_split",
    "model_training",
    "cross_validation",
    "metrics_evaluation",
    "model_selection",
    "report_generation",
]


class PipelineStateService:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, csv_path: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        job = {
            "job_id": job_id,
            "job_status": "created",
            "csv_path": csv_path,
            "model_path": None,
            "report_path": None,
            "eda_path": None,
            "steps": [{"name": step, "status": "Pending", "message": ""} for step in PIPELINE_STEPS],
            "created_at": now,
            "updated_at": now,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return deepcopy(job)

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return deepcopy(job) if job else None

    def update_job_status(self, job_id: str, status: str, error: str | None = None) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["job_status"] = status
            job["error"] = error
            job["updated_at"] = datetime.now(timezone.utc)
            return deepcopy(job)

    def update_step_status(self, job_id: str, step_name: str, status: str, message: str = "") -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for step in job["steps"]:
                if step["name"] == step_name:
                    step["status"] = status
                    step["message"] = message
                    break
            job["updated_at"] = datetime.now(timezone.utc)
            return deepcopy(job)


pipeline_state_service = PipelineStateService()
