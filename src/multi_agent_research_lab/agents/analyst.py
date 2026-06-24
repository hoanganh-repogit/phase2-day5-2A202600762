"""Analyst agent skeleton."""

import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("AnalystAgent analyzing research notes...")
        
        llm = LLMClient()
        
        system_prompt = (
            "You are the Analyst Agent of a Multi-Agent Research System.\n"
            "Your task is to analyze raw research notes and extract structured insights.\n"
            "Specifically:\n"
            "1. Extract key claims and technical facts.\n"
            "2. Compare different viewpoints, methodologies, or state-of-the-art architectures if relevant.\n"
            "3. Assess the strength of the evidence (e.g. check for citations, identify any weak claims or logical gaps).\n"
            "4. Organize findings clearly with sections and bullet points. Preserve source references."
        )
        
        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Research Notes:\n{state.research_notes or 'No research notes available.'}"
        )
        
        try:
            response = llm.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            
            # Record AgentResult
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes or "",
                    metadata={
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analysis_completed", {})
        except Exception as e:
            logger.exception("Failed to compile analysis notes")
            state.errors.append(f"AnalystAgent error: {e}")
            state.analysis_notes = f"Failed to compile analysis notes due to: {e}"
            
        return state
