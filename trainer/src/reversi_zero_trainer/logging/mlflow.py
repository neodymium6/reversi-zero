"""MLflow logger implementation."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

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
        if cfg.run_id is None:
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
            self._persist_run_id()
        else:
            run = self.client.get_run(cfg.run_id)
            experiment = self.client.get_experiment(run.info.experiment_id)
            if experiment.name != cfg.experiment_name:
                raise ValueError(
                    "MLflow run belongs to a different experiment: "
                    f"expected={cfg.experiment_name!r}, actual={experiment.name!r}"
                )
            self.run_id = cfg.run_id
            self.client.set_terminated(self.run_id, status="RUNNING")
            self.client.log_batch(
                self.run_id,
                tags=[RunTag(key, value) for key, value in cfg.tags.items()],
            )
        self._finished = False

    def _persist_run_id(self) -> None:
        path = self.cfg.run_id_path
        if path is None:
            return
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing != self.run_id:
                raise RuntimeError(
                    f"Refusing to overwrite a different MLflow run ID: {path}"
                )
            return
        next_path = path.with_name(f".{path.name}.next")
        try:
            next_path.write_text(f"{self.run_id}\n", encoding="utf-8")
            next_path.replace(path)
        finally:
            next_path.unlink(missing_ok=True)

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
        self._terminate("FINISHED")

    def fail(self) -> None:
        self._terminate("FAILED")

    def _terminate(self, status: str) -> None:
        if self._finished:
            return
        self.client.set_terminated(self.run_id, status=status)
        self._finished = True
