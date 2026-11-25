"""Fail-closed policy for optional model artifact resolution.

The policy mirrors the repository's LOCAL_ONLY / EXPLICIT_REMOTE /
REMOTE_DISABLED vocabulary. It governs model-file resolution only; it does
not authorize sending user documents or images to a remote inference service.
"""
from dataclasses import dataclass
from typing import Literal

ModelAccessMode = Literal["LOCAL_ONLY", "EXPLICIT_REMOTE", "REMOTE_DISABLED"]
_ALLOWED_MODES = {"LOCAL_ONLY", "EXPLICIT_REMOTE", "REMOTE_DISABLED"}


@dataclass(frozen=True)
class ModelAccessPolicy:
    """Require explicit authorization before a model loader may fetch remotely."""

    mode: ModelAccessMode = "LOCAL_ONLY"
    remote_authorized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in _ALLOWED_MODES:
            raise ValueError(f"Unsupported model access mode: {self.mode}")
        if not isinstance(self.remote_authorized, bool):
            raise ValueError("remote_authorized must be a boolean")
        if self.mode == "EXPLICIT_REMOTE" and not self.remote_authorized:
            raise ValueError("EXPLICIT_REMOTE requires remote_authorized=True")
        if self.mode != "EXPLICIT_REMOTE" and self.remote_authorized:
            raise ValueError("remote_authorized is valid only with EXPLICIT_REMOTE")

    @property
    def local_files_only(self) -> bool:
        """Whether model resolution must use the local cache and never fetch."""
        return self.mode != "EXPLICIT_REMOTE"


DEFAULT_MODEL_ACCESS_POLICY = ModelAccessPolicy()
