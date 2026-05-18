from typing import Any, Dict

import pandas as pd

from .base_agent import BaseAgent

try:
    from backend.services.model_store import load_model_artifacts
except ModuleNotFoundError:
    from services.model_store import load_model_artifacts


class PredictionAgent(BaseAgent):
    name = "prediction"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        job_id = context.get("job_id")
        record = context.get("record")
        if not job_id:
            raise ValueError("PredictionAgent requires job_id in context.")
        if not isinstance(record, dict):
            raise ValueError("PredictionAgent requires record dict in context.")

        artifacts = load_model_artifacts(job_id)
        pipeline = artifacts["pipeline"]
        metadata = artifacts["metadata"]
        problem_type = metadata.get("problem_type", "")
        feature_columns = metadata.get("feature_columns", [])

        input_record = {col: record.get(col, None) for col in feature_columns}
        input_df = pd.DataFrame([input_record])

        prediction = pipeline.predict(input_df)[0]
        prediction_value = prediction.item() if hasattr(prediction, "item") else prediction

        response: Dict[str, Any] = {
            "prediction": prediction_value,
            "problem_type": problem_type,
            "model_path": artifacts["model_path"],
            "explanation": {
                "model_used": metadata.get("best_model"),
                "target_column": metadata.get("selected_target"),
                "feature_columns_expected": feature_columns,
                "input_columns_received": sorted(list(record.keys())),
                "missing_features_filled_as_null": [col for col in feature_columns if col not in record],
                "how_prediction_is_done": [
                    "Input record is aligned to training feature schema.",
                    "Saved preprocessing pipeline applies imputation/encoding/scaling exactly as in training.",
                    "Trained final model predicts on transformed features.",
                ],
            },
        }

        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)[0]
            classes = pipeline.named_steps["model"].classes_
            response["probabilities"] = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probs)
            }
            predicted_label = str(prediction_value)
            response["explanation"]["probability_interpretation"] = (
                f"Predicted class '{predicted_label}' has probability "
                f"{response['probabilities'].get(predicted_label, 0.0):.4f}."
            )

        return response
