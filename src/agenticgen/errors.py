"""Failures at component boundaries; none implies a successful optimization."""


class AgenticGenError(Exception):
    """Base exception for actionable configuration and execution failures."""


class DependencyError(AgenticGenError):
    """An external implementation is unavailable or incompatible."""


class EditRejected(AgenticGenError):
    """An edit does not satisfy the declared graph contract."""


class ArtifactError(AgenticGenError):
    """An input or output artifact fails integrity validation."""
