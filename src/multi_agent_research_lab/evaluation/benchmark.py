"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def calculate_citation_coverage(final_answer: str | None, sources: list) -> float:
    """Calculate the ratio of cited sources in final_answer to total sources."""
    if not final_answer or not sources:
        return 0.0
    
    cited_count = 0
    final_answer_lower = final_answer.lower()
    for i, src in enumerate(sources):
        marker_num = f"[{i+1}]"
        # Check if the citation index [X] or url is in the text
        if marker_num in final_answer:
            cited_count += 1
        elif src.url and src.url.lower() in final_answer_lower:
            cited_count += 1
        elif len(src.title) > 5 and src.title.lower()[:20] in final_answer_lower:
            cited_count += 1
            
    return cited_count / len(sources)


def evaluate_quality_score(query: str, final_answer: str | None) -> float:
    """Use LLM-as-a-judge to evaluate response quality from 0 to 10."""
    if not final_answer:
        return 0.0
        
    try:
        llm = LLMClient()
        system_prompt = (
            "You are an expert academic and technical evaluator.\n"
            "Given the user's research query and the final report, rate the report quality on a scale from 0.0 (worst) to 10.0 (best).\n"
            "Consider:\n"
            "- Did it answer the query fully?\n"
            "- Is the structure logical and writing style professional?\n"
            "- Are sources referenced appropriately?\n\n"
            "Return ONLY the floating point number (e.g., '8.5'), with no explanation or surrounding text."
        )
        user_prompt = f"Query: {query}\n\nReport:\n{final_answer}"
        response = llm.complete(system_prompt, user_prompt)
        score = float(response.content.strip())
        return max(0.0, min(10.0, score))
    except Exception as e:
        logger.warning(f"Failed to calculate LLM-as-a-judge quality score: {e}. Defaulting to 7.0.")
        return 7.0


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, and citation coverage of a runner."""
    logger.info(f"Starting benchmark run: {run_name}...")
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # Calculate costs and tokens from agent results
    total_cost = 0.0
    total_tokens = 0
    for res in state.agent_results:
        if res.metadata:
            total_cost += res.metadata.get("cost_usd", 0.0) or 0.0
            tokens_in = res.metadata.get("tokens_in", 0) or 0
            tokens_out = res.metadata.get("tokens_out", 0) or 0
            total_tokens += tokens_in + tokens_out

    # Calculate citation coverage
    citation_cov = calculate_citation_coverage(state.final_answer, state.sources)

    # Evaluate quality
    quality = evaluate_quality_score(query, state.final_answer)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0.0 else None,
        quality_score=quality,
        citation_coverage=citation_cov,
        error_count=len(state.errors),
        total_tokens=total_tokens if total_tokens > 0 else None,
        notes=f"Processed in {state.iteration} routing iterations.",
    )
    
    logger.info(f"Benchmark run {run_name} completed. Latency: {latency:.2f}s, Cost: {total_cost:.5f} USD.")
    return state, metrics
