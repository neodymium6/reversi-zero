"""Tests for the logging system."""

import pytest

from reversi_zero_trainer.logging import (
    BaseLogger,
    ConsoleConfig,
    ListLogger,
    LoggerKind,
    LoggingConfig,
    MLflowConfig,
    create_logger,
    log_training_metrics,
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


def test_training_metric_names_are_stable_across_iterations():
    from unittest.mock import Mock

    logger = Mock(spec=BaseLogger)
    log_training_metrics(
        logger,
        {"loss/total": 2.5, "eval/value_mae": 0.4},
        step=7,
    )

    logger.log_metrics.assert_called_once_with(
        {"train/loss/total": 2.5, "train/eval/value_mae": 0.4},
        step=7,
        color="cyan",
    )


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


def test_console_and_mlflow_log_simultaneously(tmp_path, monkeypatch):
    """A backend list fans every event out to console and MLflow."""
    from mlflow import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    config = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(verbose=False),
            LoggerKind.MLFLOW: MLflowConfig(
                tracking_uri=tracking_uri,
                artifact_location=(tmp_path / "artifacts").as_uri(),
                experiment_name="multi-backend-test",
                run_name="test-run",
            ),
        }
    )

    with create_logger(config) as logger:
        assert isinstance(logger, ListLogger)
        from reversi_zero_trainer.logging.mlflow import MLflowLogger

        mlflow_logger = next(
            backend for backend in logger.backends if isinstance(backend, MLflowLogger)
        )
        batch_calls = []
        original_log_batch = mlflow_logger.client.log_batch

        def record_log_batch(*args, **kwargs):
            batch_calls.append((args, kwargs))
            return original_log_batch(*args, **kwargs)

        monkeypatch.setattr(mlflow_logger.client, "log_batch", record_log_batch)
        logger.log_params({"batch_size": 64, "epochs": 1})
        logger.log_metrics({"loss": 0.25, "accuracy": 0.75}, step=3)
        logger.log_artifact("checkpoint", str(artifact))

        assert len(batch_calls) == 2
        assert len(batch_calls[0][1]["params"]) == 2
        assert len(batch_calls[1][1]["metrics"]) == 2

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("multi-backend-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["batch_size"] == "64"
    assert run.data.params["epochs"] == "1"
    assert run.data.metrics["loss"] == pytest.approx(0.25)
    assert run.data.metrics["accuracy"] == pytest.approx(0.75)
    artifacts = client.list_artifacts(run.info.run_id, "checkpoint")
    assert [item.path for item in artifacts] == ["checkpoint/checkpoint.pt"]


def test_mlflow_logger_rejects_wrong_config_type():
    from reversi_zero_trainer.logging.mlflow import MLflowLogger

    with pytest.raises(TypeError, match="MLflowLogger requires MLflowConfig"):
        MLflowLogger(ConsoleConfig())


def test_mlflow_logger_resumes_the_same_run(tmp_path):
    from mlflow import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'resume.db'}"
    artifact_location = (tmp_path / "resume-artifacts").as_uri()
    run_id_path = tmp_path / "mlflow_run_id"
    initial = MLflowConfig(
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        experiment_name="resume-test",
        run_name="one-run",
        run_id_path=run_id_path,
    )

    with create_logger(LoggingConfig(backends={LoggerKind.MLFLOW: initial})) as logger:
        logger.log_metrics({"train/loss/total": 2.0}, step=0)

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    resumed = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name="resume-test",
        run_name="one-run",
        tags={"reversi_zero.resume": "true"},
        run_id=run_id,
        run_id_path=run_id_path,
    )
    with create_logger(LoggingConfig(backends={LoggerKind.MLFLOW: resumed})) as logger:
        logger.log_metrics({"train/loss/total": 1.0}, step=1)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("resume-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert [run.info.run_id for run in runs] == [run_id]
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.tags["reversi_zero.resume"] == "true"
    history = client.get_metric_history(run_id, "train/loss/total")
    assert [(metric.step, metric.value) for metric in history] == [(0, 2.0), (1, 1.0)]


def test_mlflow_logger_marks_failed_context(tmp_path):
    from mlflow import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'failure.db'}"
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        artifact_location=(tmp_path / "failure-artifacts").as_uri(),
        experiment_name="failure-test",
    )

    with pytest.raises(RuntimeError, match="training failed"):
        with create_logger(
            LoggingConfig(backends={LoggerKind.MLFLOW: config})
        ) as logger:
            logger.log_metrics({"train/loss/total": 2.0}, step=0)
            raise RuntimeError("training failed")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("failure-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"


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
