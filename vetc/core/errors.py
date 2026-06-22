"""Custom exception hierarchy for VET-C."""


class VETCError(Exception):
    """Base class for all VET-C domain errors."""


class TensorShapeError(VETCError):
    """Raised when a tensor does not match its expected shape."""


class TensorDTypeError(VETCError):
    """Raised when a tensor does not match its expected dtype."""


class NonFiniteTensorError(VETCError):
    """Raised when a tensor contains NaN or infinite values."""


class ConfigError(VETCError):
    """Raised when configuration values are missing or invalid."""


__all__ = [
    "ConfigError",
    "NonFiniteTensorError",
    "TensorDTypeError",
    "TensorShapeError",
    "VETCError",
]

