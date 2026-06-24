"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""
    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Tokens | Errors | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10.0"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage * 100:.1f}%"
        tokens = "" if item.total_tokens is None else f"{item.total_tokens:,}"
        errors = f"{item.error_count}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {citation} | {tokens} | {errors} | {item.notes} |"
        )
    return "\n".join(lines) + "\n"
