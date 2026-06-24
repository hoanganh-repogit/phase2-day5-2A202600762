"""Writer agent skeleton."""

import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("WriterAgent generating final answer report...")
        
        llm = LLMClient()
        
        # Get latest critic feedback
        critic_feedback = None
        for result in reversed(state.agent_results):
            if result.agent == AgentName.CRITIC:
                critic_feedback = result.content
                break
                
        system_prompt = (
            "You are the Writer Agent of a Multi-Agent Research System.\n"
            f"Your task is to produce a comprehensive, structured report tailored to: {state.request.audience}.\n"
            "Requirements:\n"
            "1. Answer the query thoroughly, logically, and professionally.\n"
            "2. Integrate inline citations referring to the sources (e.g. '[1](URL)' or '[Source Name](URL)') to support key claims.\n"
            "3. Provide a 'References' or 'Sources' list at the bottom of the report.\n"
            "4. If a previous draft and critic feedback are provided, address the feedback carefully and improve the draft."
        )
        
        sources_text = ""
        for i, src in enumerate(state.sources):
            sources_text += f"[{i+1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"
            
        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Source Documents:\n{sources_text or 'N/A'}\n\n"
            f"Previous Draft: {state.final_answer or 'None'}\n\n"
            f"Latest Critic Feedback: {critic_feedback or 'None'}"
        )
        
        try:
            response = llm.complete(system_prompt, user_prompt)
            state.final_answer = response.content
            
            # Record AgentResult
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer or "",
                    metadata={
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writing_completed", {})
        except Exception as e:
            logger.exception("Failed to write final report")
            state.errors.append(f"WriterAgent error: {e}")
            state.final_answer = f"Failed to write final report due to: {e}"
            
        return state
