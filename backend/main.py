import logging
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from backend.agents.master_agent import MasterAgent
    from backend.schemas.requests import PredictRequest, RunAutoMLRequest
    from backend.schemas.responses import (
        GenericMessageResponse,
        JobResponse,
        StatusResponse,
        StubDataResponse,
    )
    from backend.services.pipeline_state import pipeline_state_service
    from backend.services.utils import MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, ensure_storage_dirs
except ModuleNotFoundError:
    from agents.master_agent import MasterAgent
    from schemas.requests import PredictRequest, RunAutoMLRequest
    from schemas.responses import (
        GenericMessageResponse,
        JobResponse,
        StatusResponse,
        StubDataResponse,
    )
    from services.pipeline_state import pipeline_state_service
    from services.utils import MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, ensure_storage_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("automl-backend")

app = FastAPI(title="Agentic AutoML POC Backend", version="0.3.0")
master_agent = MasterAgent()


@app.on_event("startup")
def startup_event() -> None:
    ensure_storage_dirs()
    logger.info("Storage directories ensured.")


@app.get("/")
def root() -> dict:
    return {"message": "Agentic AutoML backend is running."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload-csv", response_model=JobResponse)
async def upload_csv(file: UploadFile = File(...)) -> JobResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    job_id = str(uuid.uuid4())
    target_path = UPLOADS_DIR / f"{job_id}.csv"

    try:
        content = await file.read()
        target_path.write_bytes(content)
        pipeline_state_service.create_job(job_id=job_id, csv_path=str(target_path))
        logger.info("CSV uploaded: job_id=%s file=%s", job_id, file.filename)
    except Exception as exc:
        logger.exception("CSV upload failed.")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded CSV: {exc}") from exc

    return JobResponse(job_id=job_id, message="CSV uploaded successfully.")


def _run_pipeline_async(job_id: str) -> None:
    try:
        master_agent.run(job_id)
    except Exception:
        logger.exception("Background pipeline run failed: job_id=%s", job_id)


@app.post("/run-automl", response_model=GenericMessageResponse)
def run_automl(payload: RunAutoMLRequest) -> GenericMessageResponse:
    job = pipeline_state_service.get_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["job_status"] == "running":
        return GenericMessageResponse(
            job_id=payload.job_id,
            status="already_running",
            message="AutoML pipeline is already running for this job.",
        )

    worker = threading.Thread(target=_run_pipeline_async, args=(payload.job_id,), daemon=True)
    worker.start()
    logger.info("AutoML run started in background: job_id=%s", payload.job_id)

    return GenericMessageResponse(
        job_id=payload.job_id,
        status="accepted",
        message="AutoML pipeline started. Track progress via /pipeline-status/{job_id}.",
    )


@app.get("/pipeline-status/{job_id}", response_model=StatusResponse)
def pipeline_status(job_id: str) -> StatusResponse:
    job = pipeline_state_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StatusResponse(
        job_id=job["job_id"],
        job_status=job["job_status"],
        steps=job["steps"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        error=job.get("error"),
    )


@app.get("/eda-report/{job_id}", response_model=StubDataResponse)
def eda_report(job_id: str) -> StubDataResponse:
    job = pipeline_state_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    eda_data = job.get("outputs", {}).get("eda")
    if eda_data:
        return StubDataResponse(job_id=job_id, status="ready", data=eda_data)

    return StubDataResponse(
        job_id=job_id,
        status="not_ready",
        data={
            "message": "EDA report is not generated yet.",
            "report_path": str(REPORTS_DIR / f"{job_id}_eda.json"),
        },
    )


@app.get("/model-results/{job_id}", response_model=StubDataResponse)
def model_results(job_id: str) -> StubDataResponse:
    job = pipeline_state_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    results = {
        "problem_type_detection": job.get("outputs", {}).get("problem_type_detection"),
        "model_training": job.get("outputs", {}).get("model_training"),
        "cross_validation": job.get("outputs", {}).get("cross_validation"),
        "metrics_evaluation": job.get("outputs", {}).get("metrics_evaluation"),
        "model_selection": job.get("outputs", {}).get("model_selection"),
    }
    if any(value is not None for value in results.values()):
        return StubDataResponse(job_id=job_id, status="ready", data=results)

    return StubDataResponse(
        job_id=job_id,
        status="not_ready",
        data={
            "message": "Model results are not available yet.",
            "model_path": str(MODELS_DIR / f"{job_id}_pipeline.joblib"),
        },
    )


@app.post("/predict/{job_id}", response_model=StubDataResponse)
def predict(job_id: str, payload: PredictRequest) -> StubDataResponse:
    job = pipeline_state_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StubDataResponse(
        job_id=job_id,
        status="not_ready",
        data={
            "message": "Prediction endpoint remains a stub until Phase 6.",
            "received_record": payload.record,
        },
    )


@app.get("/download-model/{job_id}")
def download_model(job_id: str):
    job = pipeline_state_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    model_path = MODELS_DIR / f"{job_id}_pipeline.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model file not found yet. It will be available in later phases.",
        )

    return FileResponse(path=str(model_path), media_type="application/octet-stream", filename=model_path.name)
