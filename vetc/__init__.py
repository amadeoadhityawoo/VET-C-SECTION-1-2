"""VET-C package foundation."""

from vetc.encoder import DualPerceptionEncoder, ExternalEncoder, InternalEncoder
from vetc.observation import ObservationLayer

__version__ = "0.1.0"

__all__ = [
    "DualPerceptionEncoder",
    "ExternalEncoder",
    "InternalEncoder",
    "ObservationLayer",
    "__version__",
]
