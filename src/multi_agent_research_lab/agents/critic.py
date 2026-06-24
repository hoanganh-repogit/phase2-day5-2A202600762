"""Optional critic agent skeleton for bonus work."""

import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        logger.info("CriticAgent reviewing draft answer...")
        
        llm = LLMClient()
        
        system_prompt = (
            "You are the Critic Agent of a Multi-Agent Research System.\n"
            "Your task is to review the draft of the final answer against the original source documents.\n"
            "Evaluate:\n"
            "1. Did the writer answer the original query thoroughly?\n"
            "2. Are the inline citations accurate and present? Are there claims without evidence?\n"
            "3. Are the facts consistent with the source snippets?\n\n"
            "Output formatting rules:\n"
            "- If the report is highly accurate, contains citations, and answers the query fully: Start your response with the word 'APPROVED' and explain why.\n"
            "- If there are gaps, missing citations, or inaccuracies: Start your response with the word 'REJECTED' followed by concrete feedback and action items for the writer."
        )
        
        sources_text = ""
        for i, src in enumerate(state.sources):
            sources_text += f"[{i+1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"
            
        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Source Documents:\n{sources_text or 'None'}\n\n"
            f"Draft Answer under review:\n{state.final_answer or 'None'}"
        )
        
        try:
            response = llm.complete(system_prompt, user_prompt)
            critic_text = response.content
            
            # Record AgentResult
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=critic_text or "",
                    metadata={
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "approved": "approved" in critic_text.lower(),
                    },
                )
            )
            state.add_trace_event("critic_completed", {"approved": "approved" in critic_text.lower()})
        except Exception as e:
            logger.exception("Failed to review final answer in CriticAgent")
            state.errors.append(f"CriticAgent error: {e}")
            
        return state
