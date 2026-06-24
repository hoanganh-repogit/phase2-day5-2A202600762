"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline_workflow(state: ResearchState) -> ResearchState:
    """Executes a real single-agent RAG pipeline."""
    from multi_agent_research_lab.services.search_client import SearchClient
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.core.schemas import AgentResult, AgentName
    
    # 1. Search
    searcher = SearchClient()
    sources = searcher.search(state.request.query, max_results=state.request.max_sources)
    state.sources = sources
    
    # 2. Answer via LLM
    llm = LLMClient()
    system_prompt = (
        "You are an expert single-agent research assistant. Given the user's research request "
        "and the retrieved search sources, produce a comprehensive, structured response answering "
        "the query. You must include inline citations using [1](URL) or similar format, and list references at the bottom."
    )
    
    sources_text = ""
    for i, src in enumerate(sources):
        sources_text += f"[{i+1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"
        
    user_prompt = (
        f"User Query: {state.request.query}\n"
        f"Sources:\n{sources_text or 'No sources available.'}"
    )
    
    response = llm.complete(system_prompt, user_prompt)
    state.final_answer = response.content
    state.record_route("done")
    
    # Save result metadata
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer,
            metadata={
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        )
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline RAG pipeline."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    with console.status("[bold green]Running Single-Agent Baseline..."):
        state = run_baseline_workflow(state)
        
    console.print(Panel(state.final_answer or "", title="Single-Agent Baseline Answer"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    
    try:
        with console.status("[bold green]Running Multi-Agent Workflow..."):
            result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
        
    console.print(Panel(result.final_answer or "", title="Multi-Agent Answer"))
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output_path: Annotated[str, typer.Option("--output", "-o", help="Output path for report")] = "reports/benchmark_report.md",
) -> None:
    """Run benchmark comparing single-agent baseline and multi-agent workflow."""
    _init()
    
    import os
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    
    console.print("[bold yellow]1. Running Single-Agent Baseline...[/bold yellow]")
    def baseline_runner(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        return run_baseline_workflow(st)
    base_state, base_metrics = run_benchmark("baseline", query, baseline_runner)
    
    console.print("[bold yellow]2. Running Multi-Agent Workflow...[/bold yellow]")
    def multi_agent_runner(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        workflow = MultiAgentWorkflow()
        return workflow.run(st)
    ma_state, ma_metrics = run_benchmark("multi-agent", query, multi_agent_runner)
    
    # Render report
    report = render_markdown_report([base_metrics, ma_metrics])
    console.print(Panel(report, title="Benchmark Result"))
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    console.print(f"[bold green]Benchmark report successfully saved to {output_path}[/bold green]")


if __name__ == "__main__":
    app()
