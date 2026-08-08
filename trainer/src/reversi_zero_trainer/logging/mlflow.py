"""MLflow logger implementation."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param

from .base import BaseLogger, register_logger
from .config import BaseLoggerConfig, LoggerKind, MLflowConfig


def _artifact_directory(name: str) -> str:
    """Convert a human-readable artifact description into a stable path."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return normalized or "artifacts"


@register_logger(LoggerKind.MLFLOW)
class MLflowLogger(BaseLogger):
    """Log metrics, parameters, and artifacts through ``MlflowClient``."""

    def __init__(self, cfg: BaseLoggerConfig) -> None:
        if not isinstance(cfg, MLflowConfig):
            raise TypeError(
                f"MLflowLogger requires MLflowConfig, but got {type(cfg).__name__}"
            )
        self.cfg = cfg
        self.client = MlflowClient(tracking_uri=cfg.tracking_uri)
        experiment = self.client.get_experiment_by_name(cfg.experiment_name)
        experiment_id = (
            self.client.create_experiment(
                cfg.experiment_name,
                artifact_location=cfg.artifact_location,
            )
            if experiment is None
            else experiment.experiment_id
        )
        run = self.client.create_run(
            experiment_id=experiment_id,
            run_name=cfg.run_name,
            tags=cfg.tags,
        )
        self.run_id = run.info.run_id
        self._finished = False

    def log_metric(
        self, name: str, value: float, step: int | None = None, color: str | None = None
    ) -> None:
        self.log_metrics({name: value}, step=step, color=color)

    def log_param(self, key: str, value: Any) -> None:
        self.log_params({key: value})

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
        color: str | None = None,
    ) -> None:
        del color
        if not metrics:
            return
        timestamp = int(time.time() * 1000)
        resolved_step = 0 if step is None else step
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(name, float(value), timestamp, resolved_step)
                for name, value in metrics.items()
            ],
        )

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not params:
            return
        self.client.log_batch(
            self.run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
        )

    def log_artifact(self, name: str, path: str) -> None:
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Artifact does not exist or is not a file: {artifact}"
            )
        self.client.log_artifact(
            self.run_id,
            str(artifact),
            artifact_path=_artifact_directory(name),
        )

    def finish(self) -> None:
        if self._finished:
            return
        self.client.set_terminated(self.run_id, status="FINISHED")
        self._finished = True
