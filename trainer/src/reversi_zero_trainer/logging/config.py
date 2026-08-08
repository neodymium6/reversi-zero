"""Configuration classes for the logging system."""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LoggerKind(str, Enum):
    """Enum for logger backend types."""

    CONSOLE = "console"
    MLFLOW = "mlflow"


class BaseLoggerConfig(ABC):
    """Base class for all logger configurations."""

    pass


@dataclass
class ConsoleConfig(BaseLoggerConfig):
    """Configuration for console logger.

    Args:
        verbose: Whether to print verbose output
        show_params_table: Whether to show parameters in a table format
        show_timestamp: Whether to show timestamps with metrics
    """

    verbose: bool = True
    show_params_table: bool = True
    show_timestamp: bool = False


@dataclass
class MLflowConfig(BaseLoggerConfig):
    """Configuration for an MLflow tracking backend."""

    tracking_uri: str | None = None
    artifact_location: str | None = None
    experiment_name: str = "reversi-zero"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None
    run_id_path: Path | None = None


@dataclass
class LoggingConfig:
    """Overall logging configuration.

    Args:
        backends: Dictionary mapping logger kinds to their configurations
    """

    backends: dict[LoggerKind, BaseLoggerConfig]
