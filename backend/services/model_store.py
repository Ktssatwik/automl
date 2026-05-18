import json
from pathlib import Path
from typing import Any, Dict

import joblib

try:
    from backend.services.utils import MODELS_DIR
except ModuleNotFoundError:
    from services.utils import MODELS_DIR
    


def model_file_path(job_id: str) -> Path:
    return MODELS_DIR / f"{job_id}_pipeline.joblib"


def metadata_file_path(job_id: str) -> Path:
    return MODELS_DIR / f"{job_id}_metadata.json"


def save_model_artifacts(job_id: str, pipeline_obj: Any, metadata: Dict[str, Any]) -> Dict[str, str]:
    model_path = model_file_path(job_id)
    metadata_path = metadata_file_path(job_id)

    joblib.dump(pipeline_obj, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }
