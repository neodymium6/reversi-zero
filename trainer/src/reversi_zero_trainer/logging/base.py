"""Base classes and registry for the logging system."""

from abc import ABC, abstractmethod
from typing import Any

from .config import BaseLoggerConfig, LoggerKind, LoggingConfig

# Global registry mapping LoggerKind to Logger classes
LOGGER_REGISTRY: dict[LoggerKind, type["BaseLogger"]] = {}


def register_logger(kind: LoggerKind):
    """Decorator to register a logger class with a specific kind.

    Args:
        kind: The LoggerKind this logger handles

    Returns:
        Decorator function that registers the class

    Example:
        @register_logger(LoggerKind.CONSOLE)
        class ConsoleLogger(BaseLogger):
            ...
    """

    def decorator(cls: type["BaseLogger"]) -> type["BaseLogger"]:
        LOGGER_REGISTRY[kind] = cls
        return cls

    return decorator


class BaseLogger(ABC):
    """Abstract base class for all loggers.

    Supports context manager protocol for automatic cleanup:
        with create_logger(config) as logger:
            logger.log_metric("loss", 0.5)
    """

    @abstractmethod
    def __init__(self, cfg: BaseLoggerConfig) -> None:
        """Initialize the logger with configuration.

        Args:
            cfg: Logger configuration (subclass-specific type)
        """
        pass

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter/configuration value.

        Args:
            key: Parameter name
            value: Parameter value
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish logging and clean up resources."""
        pass

    def __enter__(self) -> "BaseLogger":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit context manager and clean up."""
        self.finish()


class ListLogger(BaseLogger):
    """Logger that forwards calls to multiple backend loggers.

    This allows using multiple logging backends simultaneously
    (e.g., console + ClearML).
    """

    def __init__(self, backends: list[BaseLogger]) -> None:
        """Initialize with a list of backend loggers.

        Args:
            backends: List of logger instances to forward calls to
        """
        self.backends = backends

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric to all backends."""
        for backend in self.backends:
            backend.log_metric(name, value, step)

    def log_param(self, key: str, value: Any) -> None:
        """Log parameter to all backends."""
        for backend in self.backends:
            backend.log_param(key, value)

    def finish(self) -> None:
        """Finish all backends."""
        for backend in self.backends:
            backend.finish()


def create_logger(cfg: LoggingConfig) -> BaseLogger:
    """Create a logger from configuration.

    Args:
        cfg: Logging configuration specifying backends and their configs

    Returns:
        A BaseLogger instance (either a single logger or ListLogger)

    Raises:
        RuntimeError: If no backends are specified or a backend is not registered
    """
    instances: list[BaseLogger] = []

    for kind, backend_cfg in cfg.backends.items():
        logger_cls = LOGGER_REGISTRY.get(kind)
        if logger_cls is None:
            raise RuntimeError(f"Logger '{kind.value}' is not registered.")

        # Each logger's __init__ is responsible for narrowing the config type
        instances.append(logger_cls(backend_cfg))

    if not instances:
        raise RuntimeError("At least one logger backend must be specified.")

    if len(instances) == 1:
        return instances[0]
    return ListLogger(instances)
