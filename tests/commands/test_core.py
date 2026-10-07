import pytest
from ocr_core.commands.core import (
    Command, CommandRegistry, Executor, ExecutionContext, Pipeline,
)


@pytest.fixture
def reg():
    return CommandRegistry()


def test_register_and_get(reg):
    cmd = Command(id="t.x", name="", description="",
                  params_schema={"type": "object", "properties": {}, "required": []},
                  run=lambda ctx, **kw: {"x": 1})
    reg.register(cmd)
    assert "t.x" in reg
    assert reg.get("t.x").id == "t.x"


def test_validate_required(reg):
    cmd = Command(id="t", name="", description="",
                  params_schema={"type": "object",
                                 "properties": {"path": {"type": "string"}},
                                 "required": ["path"]},
                  run=lambda ctx, **kw: {})
    with pytest.raises(ValueError, match="مطلوب"):
        cmd.validate_params({})


def test_validate_type():
    cmd = Command(id="t", name="", description="",
                  params_schema={"type": "object",
                                 "properties": {"n": {"type": "integer"}},
                                 "required": []},
                  run=lambda ctx, **kw: {})
    with pytest.raises(TypeError):
        cmd.validate_params({"n": "text"})


def test_executor_ok(reg):
    reg.register(Command(id="t", name="", description="",
                         params_schema={"type": "object", "properties": {}, "required": []},
                         run=lambda ctx, **kw: {"v": 42}))
    r = Executor(reg).execute("t", {})
    assert r["ok"] is True
    assert r["result"]["v"] == 42


def test_executor_error(reg):
    def fail(ctx, **kw):
        raise RuntimeError("تعطل")
    reg.register(Command(id="f", name="", description="",
                         params_schema={"type": "object", "properties": {}, "required": []},
                         run=fail))
    r = Executor(reg).execute("f", {})
    assert r["ok"] is False
    assert "تعطل" in r["error"]


def test_pipeline(reg):
    calls = []
    for name in ("a", "b", "c"):
        reg.register(Command(id=name, name="", description="",
                             params_schema={"type": "object", "properties": {}, "required": []},
                             run=lambda ctx, n=name, **kw: calls.append(n) or {"done": n}))
    p = Pipeline("t").add("a").add("b").add("c")
    r = Executor(reg).execute_pipeline(p)
    assert r["ok"] is True
    assert calls == ["a", "b", "c"]


def test_pipeline_stops_on_error(reg):
    reg.register(Command(id="ok", name="", description="",
                         params_schema={"type": "object", "properties": {}, "required": []},
                         run=lambda ctx, **kw: {"ok": True}))
    reg.register(Command(id="fail", name="", description="",
                         params_schema={"type": "object", "properties": {}, "required": []},
                         run=lambda ctx, **kw: (_ for _ in ()).throw(RuntimeError("X"))))
    reg.register(Command(id="never", name="", description="",
                         params_schema={"type": "object", "properties": {}, "required": []},
                         run=lambda ctx, **kw: {"never": True}))
    r = Executor(reg).execute_pipeline(Pipeline("t").add("ok").add("fail").add("never"))
    assert r["ok"] is False
    assert r["stopped_at"] == "fail"
    assert len(r["steps"]) == 2
