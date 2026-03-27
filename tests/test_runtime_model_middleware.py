"""Tests for runtime model selection middleware."""

from __future__ import annotations

from types import SimpleNamespace

from juena.server import runtime_model_middleware as middleware_module


class _FakeRequest:
    def __init__(self, *, context: object | None = None, model: object | None = None) -> None:
        self.runtime = SimpleNamespace(context=context)
        self.model = model or SimpleNamespace(streaming=False)
        self.state = {}

    def override(self, **kwargs: object) -> "_FakeRequest":
        return _FakeRequest(
            context=self.runtime.context,
            model=kwargs.get("model"),
        )


def test_runtime_model_middleware_uses_runtime_context(monkeypatch) -> None:
    created: dict[str, object] = {}
    runtime_llm = object()
    middleware = middleware_module.RuntimeModelMiddleware()

    def fake_create_llm(**kwargs: object) -> object:
        created["kwargs"] = kwargs
        return runtime_llm

    monkeypatch.setattr(middleware_module.LLMFactory, "create_llm", fake_create_llm)

    request = _FakeRequest(
        context=middleware_module.RuntimeModelContext(
            provider="blablador",
            model="minimax-test",
        ),
        model=SimpleNamespace(streaming=True),
    )

    seen: dict[str, object] = {}

    def handler(updated_request: _FakeRequest) -> str:
        seen["model"] = updated_request.model
        return "ok"

    result = middleware.wrap_model_call(request, handler)

    assert result == "ok"
    assert seen["model"] is runtime_llm
    assert created["kwargs"] == {
        "provider": "blablador",
        "model": "minimax-test",
        "temperature": 0.0,
        "streaming": True,
    }


def test_runtime_model_middleware_falls_back_to_langgraph_config(monkeypatch) -> None:
    created: dict[str, object] = {}
    runtime_llm = object()
    middleware = middleware_module.RuntimeModelMiddleware()

    monkeypatch.setattr(
        middleware_module,
        "get_config",
        lambda: {"configurable": {"provider": "openai", "model": "gpt-test"}},
    )

    def fake_create_llm(**kwargs: object) -> object:
        created["kwargs"] = kwargs
        return runtime_llm

    monkeypatch.setattr(middleware_module.LLMFactory, "create_llm", fake_create_llm)

    request = _FakeRequest()

    def handler(updated_request: _FakeRequest) -> object:
        return updated_request.model

    result = middleware.wrap_model_call(request, handler)

    assert result is runtime_llm
    assert created["kwargs"] == {
        "provider": "openai",
        "model": "gpt-test",
        "temperature": 0.0,
        "streaming": False,
    }
