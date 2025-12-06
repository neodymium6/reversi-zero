"""Configuration classes for the logging system."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum


class LoggerKind(str, Enum):
    """Enum for logger backend types."""

    CONSOLE = "console"


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
class LoggingConfig:
    """Overall logging configuration.

    Args:
        backends: Dictionary mapping logger kinds to their configurations
    """

    backends: dict[LoggerKind, BaseLoggerConfig]
