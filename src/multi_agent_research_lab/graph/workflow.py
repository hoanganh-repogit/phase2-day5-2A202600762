"""LangGraph workflow skeleton."""

import logging
from langgraph.graph import StateGraph, END

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent,
)

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def build(self) -> object:
        """Create a LangGraph graph."""
        logger.info("Building MultiAgentWorkflow LangGraph...")
        
        # Build StateGraph using ResearchState as the schema
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)
        workflow.add_node("critic", self.critic.run)
        
        # Entry point
        workflow.set_entry_point("supervisor")
        
        # Routing function based on the route history
        def route_next(state: ResearchState) -> str:
            if not state.route_history:
                logger.info("No route history found. Ending workflow.")
                return END
            next_agent = state.route_history[-1]
            logger.info(f"Routing next agent: '{next_agent}'")
            if next_agent == "done":
                return END
            return next_agent

        # Conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )
        
        # Direct edges back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")
        
        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        logger.info("Running MultiAgentWorkflow...")
        app = self.build()
        
        # Invoke the compiled graph
        result = app.invoke(state)
        
        # Parse return state type safely
        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        else:
            logger.warning(f"Unexpected graph run return type: {type(result)}. Returning input state.")
            return state
