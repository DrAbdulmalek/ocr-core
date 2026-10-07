"""نظام الأوامر الموحد — مستخلص من نمط photocraft/wordcraft."""
from .core import (
    Command, CommandRegistry, ExecutionContext,
    Executor, Pipeline, registry, command,
)

__all__ = [
    "Command", "CommandRegistry", "ExecutionContext",
    "Executor", "Pipeline", "registry", "command",
]
