"""Tests for renderer protocol and registry."""

from ww.t4dv.renderers.protocol import RendererProtocol, RendererRegistry


class DummyRenderer:
    view_id = "test.view"

    def render(self, data, **kwargs):
        return {"rendered": True, "data": data}


class TestRendererRegistry:
    def test_register_and_get(self):
        reg = RendererRegistry()
        r = DummyRenderer()
        reg.register(r)
        assert reg.get("test.view") is r
        assert reg.get("nonexistent") is None

    def test_list_views(self):
        reg = RendererRegistry()
        reg.register(DummyRenderer())
        assert "test.view" in reg.list_views()

    def test_render(self):
        reg = RendererRegistry()
        reg.register(DummyRenderer())
        result = reg.render("test.view", {"x": 1})
        assert result["rendered"] is True

    def test_render_unknown_raises(self):
        reg = RendererRegistry()
        try:
            reg.render("nope", {})
            assert False, "Should have raised"
        except KeyError:
            pass

    def test_protocol_compliance(self):
        assert isinstance(DummyRenderer(), RendererProtocol)
