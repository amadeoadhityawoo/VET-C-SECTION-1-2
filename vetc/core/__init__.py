"""Core types, constants, interfaces, and validation helpers for VET-C."""

from vetc.core.constants import *  # noqa: F403
from vetc.core.errors import (
    ConfigError,
    NonFiniteTensorError,
    TensorDTypeError,
    TensorShapeError,
    VETCError,
)
from vetc.core.interfaces import (
    BaseEncoder,
    BaseLossModule,
    BasePolicyController,
    BaseRegimeEngine,
    BaseRSSM,
    BaseRuntimeComponent,
    BaseTerrainGenerator,
    BaseVETCModule,
)
from vetc.core.tensor_checks import (
    check_batch_match,
    check_dtype,
    check_finite,
    check_shape,
)
from vetc.core.types import (
    ActionCommand,
    EncoderOutput,
    FusionOutput,
    FutureTrajectory,
    LatentState,
    LossOutput,
    MemoryOutput,
    ObservationPacket,
    ObservationTensor,
    PredictionOutput,
    RegimeTensor,
    RuntimeState,
    TerrainOutput,
    TrainingBatch,
    VETCStepOutput,
)

__all__ = [
    "ActionCommand",
    "BaseEncoder",
    "BaseLossModule",
    "BasePolicyController",
    "BaseRegimeEngine",
    "BaseRSSM",
    "BaseRuntimeComponent",
    "BaseTerrainGenerator",
    "BaseVETCModule",
    "ConfigError",
    "EncoderOutput",
    "FusionOutput",
    "FutureTrajectory",
    "LatentState",
    "LossOutput",
    "MemoryOutput",
    "NonFiniteTensorError",
    "ObservationPacket",
    "ObservationTensor",
    "PredictionOutput",
    "RegimeTensor",
    "RuntimeState",
    "TensorDTypeError",
    "TensorShapeError",
    "TerrainOutput",
    "TrainingBatch",
    "VETCError",
    "VETCStepOutput",
    "check_batch_match",
    "check_dtype",
    "check_finite",
    "check_shape",
]

