"""Eval-harness tests: rubric behavior on golden fixtures and live-run states.

All tests are offline - no LLM, Tavily, or network calls.
"""

from __future__ import annotations

import pytest

from researchmind.evals import (
    GOLDEN_SET,
    MIN_SOURCES,
    GoldenCase,
    _count_citations,
    check_citations,
    check_min_length,
    check_revision_improves,
    check_structure,
    run_eval_case,
)
from researchmind.live_evals import check_live_state
from researchmind.models import Critique, TokenUsage


def _case(**overrides: object) -> GoldenCase:
    """Clone the good golden case with overrides (for rubric meta-tests)."""
    base = GOLDEN_SET[0]  # the good solid-state battery case
    values: dict = {
        "case_id": base.case_id,
        "topic": base.topic,
        "report": base.report,
        "critique": base.critique,
        "revisions": base.revisions,
        "pre_revision_score": base.pre_revision_score,
    }
    values.update(overrides)
    return GoldenCase(**values)


class TestGoldenSet:
    def test_golden_set_is_nonempty(self) -> None:
        assert len(GOLDEN_SET) >= 3

    def test_good_cases_pass_all_checks(self) -> None:
        for case in GOLDEN_SET:
            if case.case_id == "golden/tiny-stub":
                continue
            results = run_eval_case(case)
            failed = {name: detail for name, (passed, detail) in results.items() if not passed}
            assert not failed, f"{case.case_id} failed: {failed}"

    def test_stub_fails_structure_length_and_citations(self) -> None:
        stub = next(c for c in GOLDEN_SET if c.case_id == "golden/tiny-stub")
        results = run_eval_case(stub)
        failed = {name for name, (passed, _) in results.items() if not passed}
        assert {"structure", "min_length", "citations"} <= failed


class TestRubricMeta:
    """Meta-tests: the rubric itself must catch regressions, not rubber-stamp."""

    def test_structure_detects_missing_section(self) -> None:
        report = GOLDEN_SET[0].report.replace("## Sources", "## References")
        passed, detail = check_structure(_case(report=report))
        assert not passed
        assert "Sources" in detail

    def test_length_detects_truncation(self) -> None:
        passed, detail = check_min_length(_case(report="too short"))
        assert not passed
        assert "chars" in detail

    def test_citations_detects_dead_sources_section(self) -> None:
        report = GOLDEN_SET[0].report.split("## Sources")[0] + "## Sources\n\nNone available."
        passed, detail = check_citations(_case(report=report))
        assert not passed
        assert f"< minimum {MIN_SOURCES}" in detail

    def test_revision_check_passes_on_improvement(self) -> None:
        passed, _ = check_revision_improves(_case(revisions=1, pre_revision_score=6))
        assert passed

    def test_revision_check_fails_when_score_stagnates(self) -> None:
        passed, detail = check_revision_improves(
            _case(revisions=1, pre_revision_score=9, critique=Critique(score=9))
        )
        assert not passed
        assert "did not improve" in detail

    def test_revision_check_trivially_passes_without_revisions(self) -> None:
        passed, _ = check_revision_improves(_case(revisions=0, pre_revision_score=None))
        assert passed


class TestCountCitations:
    def test_counts_numbered_urls(self) -> None:
        report = "intro\n\n## Sources\n\n1. https://a.com\n2. https://b.com\n3. https://c.com\n"
        assert _count_citations(report) == 3

    def test_ignores_prose_without_urls(self) -> None:
        report = "## Sources\n\nNo URLs were used in this report."
        assert _count_citations(report) == 0


class TestLiveStateChecks:
    """Rubric applied to fabricated final states of run_research (offline)."""

    def test_clean_first_draft_passes(self) -> None:
        report = GOLDEN_SET[0].report
        state = {
            "report": report,
            "critique": Critique(score=9),
            "history": [Critique(score=9)],
            "revision": 0,
            "usage": TokenUsage(total_tokens=1234),
        }
        results = check_live_state("topic", state)
        by_name = {name: (passed, detail) for name, passed, detail in results}
        assert by_name["pipeline_completed"][0] is True
        assert by_name["structure"][0] is True
        assert by_name["citations"][0] is True
        assert by_name["score_valid"][0] is True
        assert by_name["token_usage_tracked"][0] is True

    def test_revision_improvement_passes(self) -> None:
        state = {
            "report": GOLDEN_SET[0].report,
            "critique": Critique(score=9),
            "history": [Critique(score=6), Critique(score=9)],
            "revision": 1,
            "usage": TokenUsage(total_tokens=999),
        }
        by_name = {name: passed for name, passed, _ in check_live_state("t", state)}
        assert by_name["revision_improves_score"] is True

    def test_revision_without_improvement_fails(self) -> None:
        state = {
            "report": GOLDEN_SET[0].report,
            "critique": Critique(score=6),
            "history": [Critique(score=6), Critique(score=6)],
            "revision": 1,
            "usage": TokenUsage(total_tokens=999),
        }
        results = check_live_state("t", state)
        failed = [name for name, passed, _ in results if not passed]
        assert "revision_improves_score" in failed

    def test_error_state_fails_pipeline_check(self) -> None:
        results = check_live_state("t", {"error": "RuntimeError: LLM exploded"})
        failed = [name for name, passed, _ in results if not passed]
        assert failed == ["pipeline_completed"]

    def test_missing_critique_fails_score_check(self) -> None:
        state = {
            "report": GOLDEN_SET[0].report,
            "critique": None,
            "history": [],
            "revision": 0,
            "usage": TokenUsage(total_tokens=1),
        }
        by_name = {name: passed for name, passed, _ in check_live_state("t", state)}
        assert by_name["score_valid"] is False

    def test_missing_sources_section_fails(self) -> None:
        report = GOLDEN_SET[0].report.split("## Sources")[0]
        state = {
            "report": report,
            "critique": Critique(score=5),
            "history": [Critique(score=5)],
            "revision": 0,
            "usage": TokenUsage(total_tokens=1),
        }
        by_name = {name: passed for name, passed, _ in check_live_state("t", state)}
        assert by_name["structure"] is False

    @pytest.mark.parametrize(
        "score",
        [0, 5, 10],
    )
    def test_score_bounds_accept_edge_values(self, score: int) -> None:
        state = {
            "report": GOLDEN_SET[0].report,
            "critique": Critique(score=score),
            "history": [Critique(score=score)],
            "revision": 0,
            "usage": TokenUsage(total_tokens=1),
        }
        by_name = {name: passed for name, passed, _ in check_live_state("t", state)}
        assert by_name["score_valid"] is True
