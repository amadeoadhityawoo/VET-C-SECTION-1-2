"""Abstract interfaces for future VET-C modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vetc.core.types import (
    ActionCommand,
    EncoderOutput,
    FusionOutput,
    LatentState,
    LossOutput,
    ObservationPacket,
    ObservationTensor,
    PredictionOutput,
    RegimeTensor,
    RuntimeState,
    TerrainOutput,
    TrainingBatch,
    VETCStepOutput,
)


class BaseVETCModule(torch.nn.Module, ABC):
    """Base class for neural VET-C modules."""

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> object:
        """Run the module forward pass."""


class BaseEncoder(BaseVETCModule):
    """Interface for future observation encoders."""

    @abstractmethod
    def forward(self, observations: ObservationTensor) -> EncoderOutput:
        """Encode canonical external and internal observations."""


class BaseRSSM(BaseVETCModule):
    """Interface for future recurrent state-space models."""

    @abstractmethod
    def forward(
        self,
        fusion: FusionOutput,
        previous_state: LatentState | None = None,
    ) -> LatentState:
        """Update or infer latent dynamics from fused features."""


class BaseRegimeEngine(BaseVETCModule):
    """Interface for future regime inference modules."""

    @abstractmethod
    def forward(
        self,
        latent_state: LatentState,
        prediction: PredictionOutput,
    ) -> RegimeTensor:
        """Infer the current operating regime."""


class BaseTerrainGenerator(BaseVETCModule):
    """Interface for future terrain-generation modules."""

    @abstractmethod
    def forward(
        self,
        latent_state: LatentState,
        regime: RegimeTensor,
    ) -> TerrainOutput:
        """Generate compact terrain fields."""


class BasePolicyController(BaseVETCModule):
    """Interface for future policy/control modules."""

    @abstractmethod
    def forward(
        self,
        latent_state: LatentState,
        regime: RegimeTensor,
        terrain: TerrainOutput,
    ) -> ActionCommand:
        """Generate an action command."""


class BaseLossModule(BaseVETCModule):
    """Interface for future training loss modules."""

    @abstractmethod
    def forward(
        self,
        batch: TrainingBatch,
        step_output: VETCStepOutput,
    ) -> LossOutput:
        """Compute training losses for a batch."""


class BaseRuntimeComponent(ABC):
    """Interface for non-neural runtime components."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare runtime resources."""

    @abstractmethod
    def step(
        self,
        packet: ObservationPacket,
        state: RuntimeState | None = None,
    ) -> VETCStepOutput:
        """Run one runtime step from a raw observation packet."""

    @abstractmethod
    def teardown(self) -> None:
        """Release runtime resources."""


__all__ = [
    "BaseEncoder",
    "BaseLossModule",
    "BasePolicyController",
    "BaseRegimeEngine",
    "BaseRSSM",
    "BaseRuntimeComponent",
    "BaseTerrainGenerator",
    "BaseVETCModule",
]

