from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_runs_successfully() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    res = SupervisorAgent().run(state)
    assert len(res.route_history) == 1
    assert res.route_history[0] in ["researcher", "analyst", "writer", "critic", "done"]
