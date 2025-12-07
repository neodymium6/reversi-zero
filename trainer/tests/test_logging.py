"""Tests for the logging system."""

import pytest

from reversi_zero_trainer.logging import (
    BaseLogger,
    ConsoleConfig,
    LoggerKind,
    LoggingConfig,
    create_logger,
)


def test_console_logger_creation():
    """Test that console logger can be created."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True),
        }
    )
    logger = create_logger(config)
    assert isinstance(logger, BaseLogger)


def test_console_logger_log_metric(capsys):
    """Test that console logger logs metrics correctly."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=False),
        }
    )
    logger = create_logger(config)

    logger.log_metric("test_metric", 0.123456, step=10)

    captured = capsys.readouterr()
    # Rich adds ANSI codes, so we check for the content
    assert "test_metric" in captured.out
    assert "0.123456" in captured.out
    assert "step" in captured.out
    assert "10" in captured.out


def test_console_logger_log_metric_no_step(capsys):
    """Test that console logger logs metrics without step."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=False),
        }
    )
    logger = create_logger(config)

    logger.log_metric("test_metric", 0.5)

    captured = capsys.readouterr()
    assert "test_metric" in captured.out
    assert "0.5" in captured.out or "0.500000" in captured.out


def test_console_logger_log_param(capsys):
    """Test that console logger logs parameters correctly."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=False),
        }
    )
    logger = create_logger(config)

    logger.log_param("model_name", "resnet")

    captured = capsys.readouterr()
    assert "model_name" in captured.out
    assert "resnet" in captured.out


def test_console_logger_verbose_false(capsys):
    """Test that console logger respects verbose=False."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=False),
        }
    )
    logger = create_logger(config)

    logger.log_metric("test_metric", 0.5, step=1)
    logger.log_param("test_param", "value")

    captured = capsys.readouterr()
    assert captured.out == ""


def test_console_logger_finish():
    """Test that console logger finish() works without error."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True),
        }
    )
    logger = create_logger(config)
    logger.finish()  # Should not raise


def test_console_logger_context_manager(capsys):
    """Test that console logger works as context manager."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=False),
        }
    )

    with create_logger(config) as logger:
        logger.log_metric("test_metric", 0.5, step=1)
        logger.log_param("test_param", "value")

    captured = capsys.readouterr()
    assert "test_metric" in captured.out
    assert "test_param" in captured.out
    assert "value" in captured.out


def test_create_logger_no_backends():
    """Test that create_logger fails with no backends."""
    config = LoggingConfig(backends={})

    with pytest.raises(RuntimeError, match="At least one logger backend"):
        create_logger(config)


def test_create_logger_unknown_backend():
    """Test that create_logger fails with unknown backend."""
    # Create a mock unknown logger kind by manipulating the enum
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(),
        }
    )

    # Manually inject an invalid kind (this is a bit hacky for testing)
    from reversi_zero_trainer.logging.base import LOGGER_REGISTRY

    # Backup registry
    original_registry = LOGGER_REGISTRY.copy()

    try:
        # Clear console logger from registry
        LOGGER_REGISTRY.clear()

        with pytest.raises(RuntimeError, match="is not registered"):
            create_logger(config)
    finally:
        # Restore registry
        LOGGER_REGISTRY.clear()
        LOGGER_REGISTRY.update(original_registry)


def test_console_logger_wrong_config_type():
    """Test that ConsoleLogger rejects wrong config type."""
    from reversi_zero_trainer.logging.config import BaseLoggerConfig
    from reversi_zero_trainer.logging.console import ConsoleLogger

    class WrongConfig(BaseLoggerConfig):
        pass

    with pytest.raises(TypeError, match="ConsoleLogger requires ConsoleConfig"):
        ConsoleLogger(WrongConfig())


def test_console_logger_params_table(capsys):
    """Test that console logger shows params table."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=True),
        }
    )

    with create_logger(config) as logger:
        logger.log_param("model", "resnet")
        logger.log_param("lr", 0.001)
        # Table should show when first metric is logged
        logger.log_metric("loss", 1.5, step=0)

    captured = capsys.readouterr()
    # Check for table elements (normalize whitespace for assertion)
    output = captured.out.replace("\n", " ")
    assert "Configuration" in output and "Parameters" in output
    assert "model" in captured.out
    assert "resnet" in captured.out
    assert "lr" in captured.out
    assert "0.001" in captured.out
    assert "loss" in captured.out


def test_console_logger_timestamp(capsys):
    """Test that console logger can show timestamps."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(
                verbose=True, show_params_table=False, show_timestamp=True
            ),
        }
    )
    logger = create_logger(config)
    logger.log_metric("test", 1.0, step=1)

    captured = capsys.readouterr()
    # Just check that some time-like pattern exists (HH:MM:SS)
    import re

    assert re.search(r"\d{2}:\d{2}:\d{2}", captured.out)


def test_console_logger_param_after_metric_raises():
    """Test that logging param after metric raises an error."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=True),
        }
    )
    logger = create_logger(config)

    # Log params first
    logger.log_param("param1", "value1")

    # Log metric (this triggers params_logged = True)
    logger.log_metric("loss", 1.0, step=0)

    # Try to log param after metric - should raise
    with pytest.raises(
        RuntimeError, match="Cannot log parameter .* after metrics have been logged"
    ):
        logger.log_param("param2", "value2")


def test_console_logger_artifact_logging(capsys):
    """Test that console logger can log artifacts."""
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=True, show_params_table=False),
        }
    )
    logger = create_logger(config)

    logger.log_artifact("checkpoint", "/path/to/checkpoint.pt")
    logger.log_artifact("model", "/path/to/model.pt")

    captured = capsys.readouterr()
    assert "checkpoint" in captured.out
    assert "/path/to/checkpoint.pt" in captured.out
    assert "model" in captured.out
    assert "/path/to/model.pt" in captured.out
    assert "Saved" in captured.out
