"""Renderer protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RendererProtocol(Protocol):
    """A renderer takes event data and produces a figure or bytes."""

    view_id: str

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        """Render *data* and return a figure-like object."""
        ...


class RendererRegistry:
    """Maps view_id strings to renderer instances."""

    def __init__(self) -> None:
        self._renderers: dict[str, RendererProtocol] = {}

    def register(self, renderer: RendererProtocol) -> None:
        self._renderers[renderer.view_id] = renderer

    def get(self, view_id: str) -> RendererProtocol | None:
        return self._renderers.get(view_id)

    def list_views(self) -> list[str]:
        return sorted(self._renderers.keys())

    def render(self, view_id: str, data: dict[str, Any], **kwargs: Any) -> Any:
        renderer = self._renderers.get(view_id)
        if renderer is None:
            raise KeyError(f"Unknown view_id: {view_id}")
        return renderer.render(data, **kwargs)


_registry: RendererRegistry | None = None


def get_registry() -> RendererRegistry:
    global _registry
    if _registry is None:
        _registry = RendererRegistry()
    return _registry
