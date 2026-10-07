"""النواة — Command, Registry, Executor, Pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    id: str
    name: str
    description: str
    params_schema: dict
    run: Callable[..., dict]
    category: str = "general"
    requires: tuple[str, ...] = ()

    def validate_params(self, params: dict) -> None:
        required = self.params_schema.get("required", [])
        for key in required:
            if key not in params:
                raise ValueError(f"الأمر {self.id}: المعامل '{key}' مطلوب")
        props = self.params_schema.get("properties", {})
        for key, val in params.items():
            if key in props:
                expected = props[key].get("type")
                if expected and not self._type_ok(val, expected):
                    raise TypeError(
                        f"الأمر {self.id}: '{key}' يجب أن يكون {expected}، "
                        f"وُجد {type(val).__name__}"
                    )

    @staticmethod
    def _type_ok(val, expected: str) -> bool:
        return {
            "string": isinstance(val, str),
            "number": isinstance(val, (int, float)),
            "integer": isinstance(val, int),
            "boolean": isinstance(val, bool),
            "array": isinstance(val, list),
            "object": isinstance(val, dict),
        }.get(expected, True)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires": list(self.requires),
            "params_schema": self.params_schema,
        }


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        if command.id in self._commands:
            logger.warning("استبدال أمر موجود: %s", command.id)
        self._commands[command.id] = command

    def unregister(self, command_id: str) -> None:
        self._commands.pop(command_id, None)

    def get(self, command_id: str) -> Command:
        if command_id not in self._commands:
            raise KeyError(f"أمر غير معروف: {command_id}")
        return self._commands[command_id]

    def list(self, category: Optional[str] = None) -> list[Command]:
        cmds = list(self._commands.values())
        if category:
            cmds = [c for c in cmds if c.category == category]
        return sorted(cmds, key=lambda c: c.id)

    def categories(self) -> list[str]:
        return sorted({c.category for c in self._commands.values()})

    def __contains__(self, command_id: str) -> bool:
        return command_id in self._commands


registry = CommandRegistry()


def command(
    id: str,
    name: str = "",
    description: str = "",
    category: str = "general",
    params_schema: Optional[dict] = None,
    requires: tuple[str, ...] = (),
):
    def wrapper(fn: Callable) -> Callable:
        cmd = Command(
            id=id,
            name=name or id,
            description=description or (fn.__doc__ or "").strip(),
            category=category,
            params_schema=params_schema or {"type": "object", "properties": {}},
            run=fn,
            requires=requires,
        )
        registry.register(cmd)
        return fn
    return wrapper


@dataclass
class ExecutionContext:
    data: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def add_trace(self, command_id: str, params: dict,
                  duration_ms: float, ok: bool, error: str = ""):
        self.trace.append({
            "command": command_id,
            "params": params,
            "duration_ms": round(duration_ms, 2),
            "ok": ok,
            "error": error,
        })


class Executor:
    def __init__(self, reg: CommandRegistry = None):
        self.registry = reg or registry

    def execute(self, command_id: str, params: dict,
                context: Optional[ExecutionContext] = None,
                dry_run: bool = False) -> dict:
        cmd = self.registry.get(command_id)
        cmd.validate_params(params)
        ctx = context or ExecutionContext()

        if dry_run:
            return {"dry_run": True, "would_execute": command_id, "params": params}

        start = time.perf_counter()
        try:
            result = cmd.run(ctx=ctx, **params)
            duration = (time.perf_counter() - start) * 1000
            ctx.add_trace(command_id, params, duration, ok=True)
            return {
                "ok": True,
                "command": command_id,
                "duration_ms": round(duration, 2),
                "result": result,
                "context_keys": list(ctx.data.keys()),
            }
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            ctx.add_trace(command_id, params, duration, ok=False, error=str(e))
            logger.exception("فشل الأمر %s", command_id)
            return {
                "ok": False, "command": command_id,
                "duration_ms": round(duration, 2), "error": str(e),
            }

    def execute_pipeline(self, pipeline: "Pipeline",
                         context: Optional[ExecutionContext] = None) -> dict:
        ctx = context or ExecutionContext()
        results = []
        for step in pipeline.steps:
            r = self.execute(step["command"], step["params"], ctx)
            results.append(r)
            if not r["ok"]:
                return {
                    "ok": False, "stopped_at": step["command"],
                    "steps": results, "trace": ctx.trace,
                }
        return {
            "ok": True, "steps": results,
            "context_keys": list(ctx.data.keys()), "trace": ctx.trace,
        }


class Pipeline:
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: list[dict] = []

    def add(self, command_id: str, **params) -> "Pipeline":
        self.steps.append({"command": command_id, "params": params})
        return self

    @classmethod
    def from_json(cls, data: dict) -> "Pipeline":
        p = cls(name=data.get("name", "pipeline"))
        p.steps = data.get("steps", [])
        return p

    def to_json(self) -> dict:
        return {"name": self.name, "steps": self.steps}
