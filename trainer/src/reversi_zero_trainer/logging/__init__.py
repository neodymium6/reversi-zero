"""Logging system for reversi-zero trainer.

This package provides a flexible logging abstraction that supports
multiple backends (console, ClearML, etc.) through a unified interface.

Example:
    >>> from reversi_zero_trainer.logging import (
    ...     create_logger,
    ...     LoggingConfig,
    ...     ConsoleConfig,
    ...     LoggerKind,
    ... )
    >>>
    >>> config = LoggingConfig(
    ...     backends={
    ...         LoggerKind.CONSOLE: ConsoleConfig(verbose=True),
    ...     }
    ... )
    >>> logger = create_logger(config)
    >>> logger.log_param("model", "resnet")
    >>> logger.log_metric("accuracy", 0.95, step=100)
    >>> logger.finish()
"""

from .base import BaseLogger, ListLogger, create_logger
from .config import BaseLoggerConfig, ConsoleConfig, LoggerKind, LoggingConfig

# Import to trigger @register_logger decorators
from . import console  # noqa: F401

__all__ = [
    "BaseLogger",
    "ListLogger",
    "create_logger",
    "BaseLoggerConfig",
    "LoggingConfig",
    "ConsoleConfig",
    "LoggerKind",
]
