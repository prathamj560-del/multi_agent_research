"""Tests for the LangGraph pipeline: loop routing, usage accumulation, failure paths."""

from __future__ import annotations

import pytest

from researchmind import graph as graph_module
from researchmind.graph import should_revise
from researchmind.models import Critique, TokenUsage


def _state(**overrides: object) -> dict:
    base: dict = {"topic": "test topic", "revision": 0, "history": []}
    base.update(overrides)
    return base


class TestRevisionLoopRouting:
    def test_routes_to_done_when_score_meets_threshold(self, settings) -> None:
        assert should_revise(_state(critique=Critique(score=8))) == "done"

    def test_routes_to_revise_when_score_below_threshold(self, settings) -> None:
        assert should_revise(_state(critique=Critique(score=5))) == "revise"

    def test_routes_to_done_when_revision_budget_exhausted(self, settings) -> None:
        # max_revisions=1 in test settings; one revision already consumed.
        assert should_revise(_state(critique=Critique(score=4), revision=1)) == "done"

    def test_routes_to_done_without_critique(self, settings) -> None:
        assert should_revise(_state()) == "done"


class TestNodes:
    async def test_discover_stores_summary_urls_and_usage(
        self, settings, fake_tavily, mock_transport, monkeypatch
    ) -> None:
        async def fake_gather(topic: str):
            return "summary text", ["https://example1.com/article"], TokenUsage(total_tokens=42)

        monkeypatch.setattr(graph_module, "gather_research", fake_gather)
        result = await graph_module.discover(_state(topic="batteries"))

        assert result["discovery_summary"] == "summary text"
        assert result["source_urls"] == ["https://example1.com/article"]
        assert result["usage"].total_tokens == 42

    async def test_write_applies_critique_on_revision(self, settings, monkeypatch) -> None:
        captured: dict = {}

        async def fake_write_report(topic, research, critique=None, attempt=1):
            captured["critique"] = critique
            captured["attempt"] = attempt
            return "revised report", TokenUsage(total_tokens=10)

        monkeypatch.setattr(graph_module, "write_report", fake_write_report)
        critique = Critique(score=6, improvements=["add sources"])
        result = await graph_module.write(_state(revision=1, critique=critique))

        assert result["report"] == "revised report"
        assert captured["critique"] is critique
        assert captured["attempt"] == 1

    async def test_write_skips_critique_on_first_draft(self, settings, monkeypatch) -> None:
        captured: dict = {}

        async def fake_write_report(topic, research, critique=None, attempt=1):
            captured["critique"] = critique
            return "first draft", TokenUsage()

        monkeypatch.setattr(graph_module, "write_report", fake_write_report)
        await graph_module.write(_state(revision=0, critique=Critique(score=3)))
        assert captured["critique"] is None

    async def test_critique_appends_history(self, settings, monkeypatch) -> None:
        async def fake_critique_report(report: str):
            return Critique(score=9, verdict="great"), TokenUsage(total_tokens=5)

        monkeypatch.setattr(graph_module, "critique_report", fake_critique_report)
        result = await graph_module.critique(_state(report="draft"))

        assert result["critique"].score == 9
        assert len(result["history"]) == 1

    async def test_revise_increments_counter(self, settings) -> None:
        assert graph_module.revise(_state(revision=1))["revision"] == 2


class TestRunResearch:
    async def test_returns_error_state_when_pipeline_raises(self, settings, monkeypatch) -> None:
        class FakeGraph:
            async def ainvoke(self, state, config=None):
                raise RuntimeError("LLM exploded")

        monkeypatch.setattr(graph_module, "build_graph", lambda on_step=None: FakeGraph())
        result = await graph_module.run_research("any topic")

        assert result["error"].startswith("RuntimeError")
        assert "wall_time_s" in result
        assert result["topic"] == "any topic"

    async def test_success_state_includes_wall_time(self, settings, monkeypatch) -> None:
        class FakeGraph:
            async def ainvoke(self, state, config=None):
                return {
                    "topic": state["topic"],
                    "revision": 0,
                    "usage": TokenUsage(),
                    "history": [],
                }

        monkeypatch.setattr(graph_module, "build_graph", lambda on_step=None: FakeGraph())
        result = await graph_module.run_research("topic")

        assert result["topic"] == "topic"
        assert "wall_time_s" in result

    def test_build_graph_compiles_real_graph(self, settings) -> None:
        # End-to-end wiring check (no LLM calls at compile time).
        compiled = graph_module.build_graph()
        node_names = set(compiled.get_graph().nodes.keys())
        assert {"discover", "write", "critique", "revise"} <= node_names


@pytest.mark.parametrize(
    ("score", "revision", "expected"),
    [(9, 0, "done"), (8, 0, "done"), (7, 0, "revise"), (1, 2, "done")],
)
def test_routing_matrix(settings, score: int, revision: int, expected: str) -> None:
    state = _state(critique=Critique(score=score), revision=revision)
    assert should_revise(state) == expected
