import json
import logging
from typing import Any, Dict, List, Tuple

try:
    from backend.agents.csv_understanding_agent import CSVUnderstandingAgent
    from backend.agents.cv_agent import CVAgent
    from backend.agents.domain_agent import DomainAgent
    from backend.agents.eda_agent import EDAAgent
    from backend.agents.metrics_agent import MetricsAgent
    from backend.agents.model_selection_agent import ModelSelectionAgent
    from backend.agents.preprocessing_agent import PreprocessingAgent
    from backend.agents.problem_type_agent import ProblemTypeAgent
    from backend.agents.report_agent import ReportAgent
    from backend.agents.split_agent import SplitAgent
    from backend.agents.training_agent import TrainingAgent
    from backend.services.pipeline_state import pipeline_state_service
    from backend.services.utils import REPORTS_DIR
except ModuleNotFoundError:
    from agents.csv_understanding_agent import CSVUnderstandingAgent
    from agents.cv_agent import CVAgent
    from agents.domain_agent import DomainAgent
    from agents.eda_agent import EDAAgent
    from agents.metrics_agent import MetricsAgent
    from agents.model_selection_agent import ModelSelectionAgent
    from agents.preprocessing_agent import PreprocessingAgent
    from agents.problem_type_agent import ProblemTypeAgent
    from agents.report_agent import ReportAgent
    from agents.split_agent import SplitAgent
    from agents.training_agent import TrainingAgent
    from services.pipeline_state import pipeline_state_service
    from services.utils import REPORTS_DIR

logger = logging.getLogger("automl-master-agent")


class MasterAgent:
    def __init__(self) -> None:
        self.pipeline: List[Tuple[str, Any]] = [
            ("csv_understanding", CSVUnderstandingAgent()),
            ("problem_type_detection", ProblemTypeAgent()),
            ("domain_understanding", DomainAgent()),
            ("eda", EDAAgent()),
            ("preprocessing", PreprocessingAgent()),
            ("train_test_split", SplitAgent()),
            ("model_training", TrainingAgent()),
            ("cross_validation", CVAgent()),
            ("metrics_evaluation", MetricsAgent()),
            ("model_selection", ModelSelectionAgent()),
            ("report_generation", ReportAgent()),
        ]

    def run(self, job_id: str) -> Dict[str, Any]:
        job = pipeline_state_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        context: Dict[str, Any] = {
            "job_id": job_id,
            "csv_path": job.get("csv_path"),
            "outputs": {},
        }

        pipeline_state_service.update_job_status(job_id, "running")

        try:
            for step_name, agent in self.pipeline:
                logger.info("Starting step: %s for job_id=%s", step_name, job_id)
                pipeline_state_service.update_step_status(job_id, step_name, "Running", "Step started")

                result = agent.run(context)
                context["outputs"][step_name] = result
                pipeline_state_service.set_output(job_id, step_name, result)

                if step_name == "eda":
                    eda_path = REPORTS_DIR / f"{job_id}_eda.json"
                    eda_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    pipeline_state_service.set_paths(job_id, eda_path=str(eda_path))

                pipeline_state_service.update_step_status(job_id, step_name, "Completed", "Step completed")
                logger.info("Completed step: %s for job_id=%s", step_name, job_id)

            report_path = REPORTS_DIR / f"{job_id}_report.json"
            report_payload = {
                "job_id": job_id,
                "status": "completed",
                "steps": context["outputs"],
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
            pipeline_state_service.set_paths(job_id, report_path=str(report_path))
            pipeline_state_service.update_job_status(job_id, "completed")

            return {"job_id": job_id, "status": "completed", "report_path": str(report_path)}
        except Exception as exc:
            logger.exception("Pipeline failed for job_id=%s", job_id)
            pipeline_state_service.update_job_status(job_id, "failed", error=str(exc))
            raise
