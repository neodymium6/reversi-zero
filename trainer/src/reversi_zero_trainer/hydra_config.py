"""Typed Hydra schema for training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


class InferenceDtype(str, Enum):
    auto = "auto"
    float32 = "float32"
    float16 = "float16"


class TrainingDtype(str, Enum):
    auto = "auto"
    float32 = "float32"
    bfloat16 = "bfloat16"


class LearningRateSchedule(str, Enum):
    constant = "constant"
    wsd = "wsd"


class ModelType(str, Enum):
    dummy = "dummy"
    resnet = "resnet"


class LoggingBackend(str, Enum):
    console = "console"
    mlflow = "mlflow"


class SymmetryAugmentation(int, Enum):
    one = 1
    two = 2
    four = 4
    eight = 8


@dataclass
class RunSettings:
    root: str = "runs"
    experiment_name: str = "reversi-zero"
    name: str | None = None
    resume: bool = False
    num_iterations: int = 10
    seed: int = 0


@dataclass
class HardwareSettings:
    device: Device = Device.auto
    torch_threads: int | None = None
    inference_dtype: InferenceDtype = InferenceDtype.auto


@dataclass
class SelfPlaySettings:
    games_per_iteration: int = 512
    report_interval: int | None = None
    batch_size: int | None = None
    game_concurrency: int | None = None
    batch_timeout_ms: int = 1
    simulations: int = 400
    expansion_batch_size: int = 4
    c_puct: float = 3.0


@dataclass
class TrainingSettings:
    batch_size: int = 256
    num_workers: int | None = None
    epochs: int = 1
    replay_window: int = 5
    symmetry_augmentation: SymmetryAugmentation = SymmetryAugmentation.eight
    learning_rate: float = 0.001
    lr_schedule: LearningRateSchedule = LearningRateSchedule.wsd
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    dtype: TrainingDtype = TrainingDtype.auto


@dataclass
class ReferenceSettings:
    enabled: bool = True
    games: int = 40


@dataclass
class PromotionSettings:
    enabled: bool = True
    openings: int = 80
    opening_plies: int = 8
    seed: int = 0
    simulations: int | None = None
    c_puct: float = 1.5
    expansion_batch_size: int | None = None
    threshold: float = 0.55
    require_confidence: bool = False


@dataclass
class ModelSettings:
    type: ModelType = ModelType.dummy
    channels: int = 64
    blocks: int = 6


@dataclass
class ConsoleLoggingSettings:
    verbose: bool = True
    show_params_table: bool = True
    show_timestamp: bool = True


@dataclass
class MLflowLoggingSettings:
    tracking_uri: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LoggingSettings:
    backends: list[LoggingBackend] = field(
        default_factory=lambda: [LoggingBackend.console]
    )
    console: ConsoleLoggingSettings = field(default_factory=ConsoleLoggingSettings)
    mlflow: MLflowLoggingSettings = field(default_factory=MLflowLoggingSettings)


@dataclass
class TrainHydraConfig:
    run: RunSettings = field(default_factory=RunSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    selfplay: SelfPlaySettings = field(default_factory=SelfPlaySettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    promotion: PromotionSettings = field(default_factory=PromotionSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def register_train_config() -> None:
    """Register the schema used by the packaged Hydra configuration."""
    ConfigStore.instance().store(name="train_schema", node=TrainHydraConfig)


def _reject_unknown_fields(value: Any, path: str = "") -> None:
    if not is_dataclass(value) or isinstance(value, type):
        return
    declared = {item.name for item in fields(value)}
    extra = sorted(set(vars(value)) - declared)
    if extra:
        keys = ", ".join(f"{path}{name}" for name in extra)
        raise ValueError(f"Unknown Hydra configuration key(s): {keys}")
    for item in fields(value):
        _reject_unknown_fields(getattr(value, item.name), f"{path}{item.name}.")


def materialize_train_config(config: DictConfig) -> TrainHydraConfig:
    """Materialize and strictly validate a composed training configuration."""
    result = OmegaConf.to_object(config)
    if not isinstance(result, TrainHydraConfig):
        raise TypeError("Hydra configuration is not backed by TrainHydraConfig")
    _reject_unknown_fields(result)
    return result
