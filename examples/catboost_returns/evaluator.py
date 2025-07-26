"""Evaluator for CatBoost return prediction example."""
import importlib.util
import os
import yaml

from openevolve.evaluation_result import EvaluationResult


def _load_horizon(default: int = 1) -> int:
    """Load horizon from config.yaml in the same directory."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("horizon", default))
    return default


def evaluate(program_path: str) -> EvaluationResult:
    """Run the program's train_model and return combined_score and rmse."""
    horizon = _load_horizon()

    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "train_model"):
        return EvaluationResult(metrics={"combined_score": 0.0, "error": 1})

    try:
        result = program.train_model(horizon=horizon)
        score = float(result.get("combined_score", 0.0)) if isinstance(result, dict) else float(result)
    except Exception as e:
        print(f"Error running program: {e}")
        return EvaluationResult(metrics={"combined_score": 0.0, "error": 1})

    return EvaluationResult(metrics={"combined_score": score, "error": 0})
