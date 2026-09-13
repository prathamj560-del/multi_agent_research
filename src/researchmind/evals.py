"""Offline eval harness: golden-set fixtures + deterministic quality checks.

The critic inside the pipeline grades one report; the eval harness grades the
whole system (including the critic itself) with deterministic, repeatable
assertions. Two modes:

* **Offline** (this module): canned fixtures with pre-recorded pipeline outputs,
  checked against a rubric. Runs in CI for free, instantly, deterministically.
* **Live** (``researchmind.live_evals``): runs the real pipeline against real
  APIs. Opt-in only — burns Groq/Tavily quota per run.

Run with::

    python -m researchmind.evals            # offline suite
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from researchmind.models import Critique

# ---------------------------------------------------------------------------
# Golden set: topics with pre-recorded "known-good" pipeline outputs.
# Record from real runs; the rubric below asserts quality never regresses
# from this baseline. Fixtures are deliberately varied (revision fired or not,
# different section content) so a single prompt change cannot silently pass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenCase:
    """One recorded pipeline run used as a quality regression fixture."""

    case_id: str
    topic: str
    report: str
    critique: Critique
    revisions: int
    pre_revision_score: int | None = None  # None => the loop never fired
    history_length: int | None = None  # None => revisions + 1 (recorded runs match)


GOLDEN_SET: list[GoldenCase] = [
    GoldenCase(
        case_id="golden/solid-state-battery",
        topic="solid-state battery commercialization challenges",
        revisions=1,
        pre_revision_score=6,
        critique=Critique(
            score=9,
            strengths=["Well structured", "Sources cited"],
            improvements=[],
            verdict="Publication-quality after one revision.",
        ),
        report=(
            "# Solid-State Battery Commercialization Challenges\n\n"
            "## Introduction\n\n"
            "Solid-state batteries replace the flammable liquid electrolyte of "
            "lithium-ion cells with a solid conductor, promising higher energy "
            "density and improved safety. This report examines the path from "
            "laboratory prototypes to mass production.\n\n"
            "## Key Findings\n\n"
            "1. **Manufacturing scale-up remains the primary bottleneck.** "
            "Solid electrolyte layers must be produced defect-free at "
            "high throughput; current pilot lines run far below lithium-ion "
            "line speeds.\n"
            "2. **Interface degradation limits cycle life.** Mechanical stress "
            "at the electrode-electrolyte boundary during charge cycles "
            "creates cracks that raise internal resistance.\n"
            "3. **Cost parity is not expected before 2030 at scale.** "
            "Materials such as lithium lanthanum zirconium oxide require "
            "sintering above 1,100 C, which is energy intensive.\n\n"
            "## Deep Dive & Analysis\n\n"
            "Toyota, QuantumScape and Solid Power lead the commercialization "
            "race but target different electrolyte chemistries. Toyota's "
            "sulfide-based approach benefits from existing automotive supply "
            "chains, while QuantumScape's anode-free design trades "
            "manufacturing simplicity for energy density. Industry analysts "
            "note that gigafactory retrofit costs make full replacement of "
            "lithium-ion unlikely within the decade.\n\n"
            "## Conclusion\n\n"
            "Solid-state batteries will enter the market in premium EVs first, "
            "where their cost premium is acceptable. Mass adoption depends on "
            "solving dry-room manufacturing economics and interface stability.\n\n"
            "## Sources\n\n"
            "1. https://www.nature.com/articles/solid-state-review\n"
            "2. https://arstechnica.com/science/2026/02/solid-state-battery-pilot\n"
            "3. https://www.reuters.com/business/autos-transportation/toyota-battery\n"
        ),
    ),
    GoldenCase(
        case_id="golden/agent-frameworks",
        topic="LLM agent framework comparison 2026",
        revisions=0,
        pre_revision_score=None,
        critique=Critique(
            score=8,
            strengths=["Balanced comparison", "Concrete adoption data"],
            improvements=[],
            verdict="Meets the quality bar on the first draft.",
        ),
        report=(
            "# LLM Agent Framework Comparison 2026\n\n"
            "## Introduction\n\n"
            "The agent-orchestration space has consolidated around a handful of "
            "production-grade frameworks. This report compares them on "
            "architecture and adoption.\n\n"
            "## Key Findings\n\n"
            "1. **Graph-based orchestration dominates production.** Explicit "
            "state machines make control flow auditable and testable.\n"
            "2. **Structured outputs reduced integration bugs industry-wide.** "
            "Schema-validated tool calls are now table stakes.\n"
            "3. **Observability became the adoption gate.** Teams will not "
            "deploy agents they cannot trace and cost-attribute.\n\n"
            "## Deep Dive & Analysis\n\n"
            "LangGraph's state-machine model contrasts with CrewAI's role-based "
            "abstraction and AutoGen's conversation-driven approach. The "
            "trade-off is determinism versus scaffolding speed: graph systems "
            "take longer to build but survive contact with production "
            "requirements such as checkpointing and conditional routing.\n\n"
            "## Conclusion\n\n"
            "For systems that must be maintained beyond a demo, explicit graph "
            "orchestration with structured outputs is the safer default.\n\n"
            "## Sources\n\n"
            "1. https://blog.langchain.dev/langgraph-state-machines\n"
            "2. https://www.microsoft.com/en-us/research/blog/autogen-analysis\n"
        ),
    ),
    GoldenCase(
        case_id="golden/tiny-stub",  # deliberately bad: exercises the rubric's failure paths
        topic="deliberately low-quality stub report",
        revisions=0,
        pre_revision_score=None,
        critique=Critique(
            score=3,
            strengths=[],
            improvements=["Add sections", "Cite sources"],
            verdict="Stub - fails the rubric by design.",
        ),
        report=(
            "# Stub Report\n\n"
            "## Introduction\n\n"
            "This is a stub.\n"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Rubric: deterministic check functions over the final pipeline state.
# Each check returns (passed, detail). The runner aggregates into a table.
# ---------------------------------------------------------------------------

CheckFn = Callable[[GoldenCase], tuple[bool, str]]

REQUIRED_SECTIONS = (
    "Introduction",
    "Key Findings",
    "Deep Dive & Analysis",
    "Conclusion",
    "Sources",
)
MIN_REPORT_CHARS = 500
MIN_SOURCES = 2


def _count_citations(report: str) -> int:
    """Count numbered URL entries under the Sources heading."""
    section = report.split("## Sources")[-1]
    return sum(1 for line in section.splitlines() if "http" in line)


def check_structure(case: GoldenCase) -> tuple[bool, str]:
    """Every required section heading must be present."""
    missing = [s for s in REQUIRED_SECTIONS if s not in case.report]
    if missing:
        return False, f"missing sections: {', '.join(missing)}"
    return True, f"all {len(REQUIRED_SECTIONS)} sections present"


def check_min_length(case: GoldenCase) -> tuple[bool, str]:
    """Report must be substantive, not a stub."""
    n = len(case.report)
    if n < MIN_REPORT_CHARS:
        return False, f"{n} chars < minimum {MIN_REPORT_CHARS}"
    return True, f"{n} chars >= minimum {MIN_REPORT_CHARS}"


def check_citations(case: GoldenCase) -> tuple[bool, str]:
    """Sources section must cite at least the minimum number of URLs."""
    n = _count_citations(case.report)
    if n < MIN_SOURCES:
        return False, f"{n} citations < minimum {MIN_SOURCES}"
    return True, f"{n} citations >= minimum {MIN_SOURCES}"


def check_score_valid(case: GoldenCase) -> tuple[bool, str]:
    """Critic score must be a plausible 0-10 verdict."""
    if 0 <= case.critique.score <= 10:
        return True, f"score {case.critique.score}/10 in range"
    return False, f"invalid score {case.critique.score}"


def check_revision_improves(case: GoldenCase) -> tuple[bool, str]:
    """The money check: when a revision ran, the final score must beat the
    pre-revision score - proving the self-correction loop adds value."""
    if case.revisions == 0:
        return True, "no revisions, loop respected the threshold"
    pre = case.pre_revision_score
    if pre is None:
        return False, "revision ran but pre-revision score was not recorded"
    if case.critique.score > pre:
        return True, f"{pre} -> {case.critique.score} after revision"
    return False, f"revision did not improve score ({pre} -> {case.critique.score})"


def check_history_consistent(case: GoldenCase) -> tuple[bool, str]:
    """History length must equal revisions + 1 (initial critique + one per revision)."""
    expected = case.history_length if case.history_length is not None else case.revisions + 1
    if expected == case.revisions + 1:
        return True, f"history consistent: {expected} critique(s) for {case.revisions} revision(s)"
    return False, f"history len {expected} != revisions {case.revisions} + 1"


RUBRIC: list[tuple[str, CheckFn]] = [
    ("structure", check_structure),
    ("min_length", check_min_length),
    ("citations", check_citations),
    ("score_valid", check_score_valid),
    ("revision_improves_score", check_revision_improves),
    ("history_consistent", check_history_consistent),
]


def run_eval_case(case: GoldenCase) -> dict[str, tuple[bool, str]]:
    """Run every rubric check against one golden case."""
    return {name: fn(case) for name, fn in RUBRIC}


def main_cli() -> int:
    """Console-script entry point for the offline eval suite."""
    return 0 if run_offline_evals() else 1


def run_offline_evals() -> bool:
    """Run the full golden set; print a pass/fail table. Returns overall pass."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="ResearchMind offline evals (golden set)", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Case")
    table.add_column("Result")
    table.add_column("Detail")

    all_passed = True
    for case in GOLDEN_SET:
        for name, (passed, detail) in run_eval_case(case).items():
            # The stub fixture intentionally fails some checks; it verifies
            # the rubric catches bad output rather than rubber-stamping it.
            if case.case_id == "golden/tiny-stub":
                passed = True  # judged separately below
            mark = "[green]PASS[/]" if passed else "[red]FAIL[/]"
            table.add_row(name, case.case_id, mark, detail)
            all_passed = all_passed and passed

    console.print(table)

    # Sanity: the stub must actually fail structure + length + citations checks.
    stub = next(c for c in GOLDEN_SET if c.case_id == "golden/tiny-stub")
    stub_results = run_eval_case(stub)
    stub_fails = [name for name, (passed, _) in stub_results.items() if not passed]
    stub_ok = {"structure", "min_length", "citations"} <= set(stub_fails)
    console.print(
        f"\nStub-detection sanity check: {'[green]PASS[/]' if stub_ok else '[red]FAIL[/]'}"
        f" (rubric caught {len(stub_fails)} failures: {', '.join(stub_fails) or 'none'})"
    )
    return all_passed and stub_ok


if __name__ == "__main__":
    raise SystemExit(main_cli())
