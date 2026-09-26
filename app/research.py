"""The research engine: a LangGraph state machine.

    plan ──► research (parallel fan-out via Send) ──► verify ──┬─► research (capped follow-ups)
                                                               └─► synthesize ──► END

Loop controls
  * max_verify_passes      – how many follow-up research rounds the verifier may trigger
  * max_followup_queries   – queries per follow-up round
  * graph_recursion_limit  – LangGraph's own hard super-step ceiling
  * Budget                 – token / LLM-call / search / scrape caps; tripping any cap routes
                             straight to synthesize, which falls back to a zero-cost template.
"""
from __future__ import annotations

import operator
import re
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Settings
from .guardrails import (
    INJECTION_MARKER,
    UNTRUSTED_POLICY,
    Budget,
    BudgetExceeded,
    sanitize_untrusted,
    wrap_untrusted,
)
from .llm import LLM, parse_json
from .tools import fetch_page, web_search


def _or(a: bool, b: bool) -> bool:
    return bool(a) or bool(b)


class ResearchState(TypedDict, total=False):
    query: str
    subtasks: list[str]
    findings: Annotated[list[dict[str, Any]], operator.add]
    searched: Annotated[list[str], operator.add]
    halted: Annotated[bool, _or]
    pass_count: int
    followups: list[str]
    next_step: str
    verified_ids: list[str]
    open_issues: list[dict[str, Any]]
    verification_log: Annotated[list[dict[str, Any]], operator.add]
    stopped_reason: str
    report: str
    sources: list[dict[str, Any]]


@dataclass
class Ctx:
    settings: Settings
    llm: LLM
    budget: Budget
    max_passes: int


def _ctx(config: RunnableConfig) -> Ctx:
    return config["configurable"]["ctx"]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def plan_node(state: ResearchState, config: RunnableConfig) -> dict:
    c = _ctx(config)
    q = state["query"]
    n = c.settings.max_subtasks

    def mock() -> str:
        import json

        return json.dumps(
            {"subtasks": [f"{q} overview", f"{q} industry report", f"{q} analysis"][:n]}
        )

    try:
        raw = await c.llm.complete(
            system=(
                "You are a research planner. Break the request into 2 to "
                f"{n} independent, specific web-search queries that together answer it. "
                'Return JSON: {"subtasks": ["query", ...]}'
            ),
            user=f"Research request: {q}",
            budget=c.budget,
            max_tokens=300,
            json_mode=True,
            mock=mock,
        )
        subtasks = [str(s).strip() for s in parse_json(raw).get("subtasks", []) if str(s).strip()]
    except BudgetExceeded:
        return {"subtasks": [], "halted": True}
    return {"subtasks": (subtasks or [q])[:n], "pass_count": 0}


async def research_node(payload: dict, config: RunnableConfig) -> dict:
    """One parallel worker: search → fetch → sanitize → extract cited claims."""
    c = _ctx(config)
    task, pass_no = payload["task"], payload["pass_no"]
    try:
        results = await web_search(task, c.settings, c.budget)
    except BudgetExceeded:
        return {"halted": True, "searched": [task]}
    except Exception as exc:  # network/provider errors shouldn't kill the whole job
        c.budget.note("search_error", f"{task}: {exc}")
        return {"searched": [task]}

    sources: list[dict[str, str]] = []
    for i, res in enumerate(results):
        text = res.get("snippet", "")
        if i < c.settings.scrape_per_query:
            try:
                page = await fetch_page(res["url"], c.settings, c.budget)
                text = page.get("text") or text
            except BudgetExceeded:
                pass  # keep snippet, stop fetching
            except Exception as exc:
                c.budget.note("fetch_error", f"{res['url']}: {str(exc)[:120]}")
        clean = sanitize_untrusted(text, c.settings.max_page_chars, c.budget, res["url"])
        if clean:
            sources.append({"url": res["url"], "title": res.get("title", ""), "text": clean})

    if not sources:
        return {"searched": [task]}

    def mock() -> str:
        import json

        items = []
        for idx, s in enumerate(sources, 1):
            for line in s["text"].split("\n"):
                line = line.strip()
                if line and INJECTION_MARKER not in line and len(line) > 30:
                    items.append({"claim": line, "source": idx})
        return json.dumps({"findings": items[:8]})

    blocks = "\n\n".join(
        f"[{i}] {s['title']} ({s['url']})\n{wrap_untrusted(s['text'], s['url'])}" for i, s in enumerate(sources, 1)
    )
    try:
        raw = await c.llm.complete(
            system=(
                "You extract factual claims for a research report. Use ONLY the numbered sources. "
                "Every claim must be directly supported by the cited source; skip anything speculative. "
                'Return JSON: {"findings": [{"claim": "...", "source": <number>}]} with at most 8 items.\n'
                + UNTRUSTED_POLICY
            ),
            user=f"Sub-task: {task}\n\nSources:\n{blocks}",
            budget=c.budget,
            max_tokens=900,
            json_mode=True,
            mock=mock,
        )
    except BudgetExceeded:
        return {"halted": True, "searched": [task]}

    findings = []
    for item in parse_json(raw).get("findings", []):
        try:
            src = sources[int(item.get("source")) - 1]
        except (TypeError, ValueError, IndexError):
            continue  # claim without a valid source → dropped (unsupported)
        claim = str(item.get("claim", "")).strip()
        if claim and INJECTION_MARKER not in claim:
            findings.append(
                {
                    "id": uuid.uuid4().hex[:6],
                    "claim": claim[:500],
                    "url": src["url"],
                    "title": src["title"],
                    "subtask": task,
                    "pass_no": pass_no,
                }
            )
    return {"findings": findings, "searched": [task]}


async def verify_node(state: ResearchState, config: RunnableConfig) -> dict:
    c = _ctx(config)
    q = state["query"]
    findings = state.get("findings", [])
    pass_count = state.get("pass_count", 0) + 1
    halted = state.get("halted", False)

    verified_ids = [f["id"] for f in findings]
    issues: list[dict] = []
    followups: list[str] = []

    if not halted and findings:
        def mock() -> str:
            import json

            if pass_count == 1:
                return json.dumps(
                    {
                        "verified_ids": verified_ids,
                        "issues": [
                            {
                                "type": "gap",
                                "detail": "No evidence yet on recent developments or regulation.",
                                "finding_ids": [],
                            }
                        ],
                        "followup_queries": [f"{q} news"],
                    }
                )
            return json.dumps({"verified_ids": verified_ids, "issues": [], "followup_queries": []})

        listing = "\n".join(f"- id={f['id']} | {f['claim']} | source: {f['url']}" for f in findings)
        try:
            raw = await c.llm.complete(
                system=(
                    "You are a strict fact-checking reviewer. Given a research question and extracted findings:\n"
                    "1. List ids of findings that are specific, relevant and consistent (verified_ids).\n"
                    "2. Report issues: type 'gap' (important aspect not covered), 'contradiction' (findings "
                    "disagree), or 'unsupported' (vague, off-topic or implausible claim) with finding_ids.\n"
                    "3. Propose up to 3 targeted web-search queries that would close the gaps or resolve "
                    "contradictions. Return an empty list if coverage is sufficient.\n"
                    'Return JSON: {"verified_ids": [], "issues": [{"type": "", "detail": "", "finding_ids": []}], '
                    '"followup_queries": []}\n' + UNTRUSTED_POLICY
                ),
                user=f"Question: {q}\n\nFindings:\n{wrap_untrusted(listing, 'extracted findings')}",
                budget=c.budget,
                max_tokens=700,
                json_mode=True,
                mock=mock,
            )
            data = parse_json(raw)
            known = {f["id"] for f in findings}
            verified_ids = [i for i in data.get("verified_ids", []) if i in known] or verified_ids
            issues = [i for i in data.get("issues", []) if isinstance(i, dict)]
            already = {s.lower() for s in state.get("searched", [])}
            followups = [
                str(x).strip()
                for x in data.get("followup_queries", [])
                if str(x).strip() and str(x).strip().lower() not in already
            ][: c.settings.max_followup_queries]
        except BudgetExceeded:
            halted = True

    if halted:
        next_step, reason = "synthesize", "budget_exceeded"
    elif followups and pass_count <= c.max_passes:
        next_step, reason = "research", ""
    elif followups:
        next_step, reason = "synthesize", "max_passes_reached"
    else:
        next_step, reason = "synthesize", "verified_complete"

    log = {
        "pass": pass_count,
        "findings_reviewed": len(findings),
        "verified": len(verified_ids),
        "issues": issues,
        "followup_queries": followups if next_step == "research" else [],
        "decision": "follow-up research" if next_step == "research" else f"stop ({reason})",
    }
    return {
        "pass_count": pass_count,
        "verified_ids": verified_ids,
        "open_issues": issues,
        "followups": followups,
        "next_step": next_step,
        "stopped_reason": reason,
        "verification_log": [log],
        "halted": halted,
    }


async def synthesize_node(state: ResearchState, config: RunnableConfig) -> dict:
    c = _ctx(config)
    q = state["query"]
    vids = set(state.get("verified_ids", []))
    pool = [f for f in state.get("findings", []) if f["id"] in vids] or state.get("findings", [])
    findings, seen_claims = [], set()
    for f in pool:  # de-duplicate identical claims found by parallel workers
        key = re.sub(r"\W+", " ", f["claim"].lower()).strip()
        if key not in seen_claims:
            seen_claims.add(key)
            findings.append(f)

    sources: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for f in findings:
        if f["url"] not in index:
            index[f["url"]] = len(sources) + 1
            sources.append({"n": index[f["url"]], "url": f["url"], "title": f["title"]})
        f["n"] = index[f["url"]]

    fallback = _template_report(q, findings, state.get("subtasks", []))
    body = fallback
    if findings and not state.get("halted"):
        listing = "\n".join(f"- [{f['n']}] {f['claim']}" for f in findings)
        try:
            body = await c.llm.complete(
                system=(
                    "Write a structured markdown research report answering the question, using ONLY the "
                    "verified findings provided. Cite every factual sentence with its source number like [2]. "
                    "Never invent sources or numbers. Sections: '## Executive summary', '## Key findings', "
                    "'## Analysis', '## Open questions'. Do not add a Sources section.\n" + UNTRUSTED_POLICY
                ),
                user=f"Question: {q}\n\nVerified findings:\n{wrap_untrusted(listing, 'verified findings')}",
                budget=c.budget,
                max_tokens=1500,
                mock=lambda: fallback,
            )
        except BudgetExceeded:
            body = fallback

    # Citation guardrail: strip any [n] the model invented
    valid = {s["n"] for s in sources}
    body = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", body)

    reason = state.get("stopped_reason") or ("budget_exceeded" if state.get("halted") else "verified_complete")
    parts = [f"# Research report: {q}", "", body.strip(), ""]
    if state.get("open_issues") and reason != "verified_complete":
        parts += ["## Unresolved verification notes", ""]
        parts += [f"- **{i.get('type', 'note')}**: {i.get('detail', '')}" for i in state["open_issues"]]
        parts.append("")
    parts += ["## Sources", ""]
    parts += [f"{s['n']}. [{s['title'] or s['url']}]({s['url']})" for s in sources] or ["_No sources collected._"]
    if reason == "budget_exceeded":
        parts += ["", "> ⚠️ A budget cap was reached. This report contains partial results."]
    return {"report": "\n".join(parts), "sources": sources, "stopped_reason": reason}


def _template_report(q: str, findings: list[dict], subtasks: list[str]) -> str:
    """Zero-cost report used in mock mode and whenever the budget is exhausted."""
    if not findings:
        return "## Executive summary\n\nNo verified findings were collected for this request."
    lines = [
        "## Executive summary",
        "",
        f"This report compiles {len(findings)} verified findings from "
        f"{len({f['url'] for f in findings})} sources on: *{q}*.",
        "",
        "## Key findings",
        "",
    ]
    groups: dict[str, list[dict]] = {}
    for f in findings:
        groups.setdefault(f["subtask"], []).append(f)
    for task, items in groups.items():
        lines += [f"### {task}", ""]
        lines += [f"- {f['claim']} [{f['n']}]" for f in items]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_plan(state: ResearchState):
    if state.get("halted") or not state.get("subtasks"):
        return "synthesize"
    return [Send("research", {"task": t, "pass_no": 0}) for t in state["subtasks"]]


def route_after_verify(state: ResearchState):
    if state.get("next_step") == "research":
        return [Send("research", {"task": t, "pass_no": state["pass_count"]}) for t in state["followups"]]
    return "synthesize"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("research", research_node)
    g.add_node("verify", verify_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, ["research", "synthesize"])
    g.add_edge("research", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["research", "synthesize"])
    g.add_edge("synthesize", END)
    return g.compile()


GRAPH = build_graph()


async def run_research(
    query: str,
    settings: Settings,
    *,
    max_passes: int | None = None,
    token_budget: int | None = None,
) -> dict[str, Any]:
    """Run one research job end to end. Request overrides can only tighten server caps."""
    budget = Budget(
        token_limit=min(token_budget or settings.job_token_budget, settings.job_token_budget),
        llm_call_limit=settings.max_llm_calls,
        search_limit=settings.max_search_calls,
        scrape_limit=settings.max_scrape_calls,
    )
    passes = settings.max_verify_passes if max_passes is None else min(max_passes, settings.max_verify_passes)
    ctx = Ctx(settings=settings, llm=LLM(settings), budget=budget, max_passes=passes)
    query = sanitize_untrusted(query, 500)

    try:
        final = await GRAPH.ainvoke(
            {"query": query},
            config={"recursion_limit": settings.graph_recursion_limit, "configurable": {"ctx": ctx}},
        )
    except GraphRecursionError:
        budget.note("cap_hit:recursion_limit", f"graph exceeded {settings.graph_recursion_limit} steps")
        return {
            "query": query,
            "report": "",
            "sources": [],
            "findings": [],
            "verification_log": [],
            "subtasks": [],
            "stopped_reason": "recursion_limit",
            "usage": budget.snapshot(),
            "model": ctx.llm.model_name,
        }

    vids = set(final.get("verified_ids", []))
    return {
        "query": query,
        "report": final.get("report", ""),
        "sources": final.get("sources", []),
        "findings": [{**f, "verified": f["id"] in vids} for f in final.get("findings", [])],
        "verification_log": final.get("verification_log", []),
        "subtasks": final.get("subtasks", []),
        "stopped_reason": final.get("stopped_reason", ""),
        "usage": budget.snapshot(),
        "model": ctx.llm.model_name,
    }
