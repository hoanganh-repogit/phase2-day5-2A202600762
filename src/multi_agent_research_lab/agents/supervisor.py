"""Supervisor / router skeleton."""

import json
import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.llm_client import LLMClient


logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Uses LLM decision making with rule-based fallback and max iterations guardrails.
        """
        settings = get_settings()

        # Guardrail: Enforce max iterations
        if state.iteration >= settings.max_iterations:
            logger.warning(f"Max iterations ({settings.max_iterations}) reached. Routing to done.")
            state.record_route("done")
            state.add_trace_event("supervisor_decision", {"next_step": "done", "reason": "Max iterations reached"})
            return state

        # Collect state information
        has_research = bool(state.research_notes)
        has_analysis = bool(state.analysis_notes)
        has_final = bool(state.final_answer)

        # Get latest critic feedback
        critic_feedback = None
        for result in reversed(state.agent_results):
            if result.agent == AgentName.CRITIC:
                critic_feedback = result.content
                break

        # Rule-based default logic (also used as fallback)
        if not has_research:
            fallback_step = "researcher"
        elif not has_analysis:
            fallback_step = "analyst"
        elif not has_final:
            fallback_step = "writer"
        elif critic_feedback and "approve" not in critic_feedback.lower():
            # If critic had feedback and did not approve, go back to writer/analyst
            fallback_step = "writer"
        else:
            fallback_step = "done"

        # Ask LLM for the routing decision
        try:
            llm = LLMClient()
            system_prompt = (
                "You are the Supervisor of a Multi-Agent Research System.\n"
                "Your task is to analyze the current system state and decide the next optimal action.\n"
                "The available next steps are:\n"
                "- 'researcher': If we need to gather information or if the current research notes are missing/insufficient.\n"
                "- 'analyst': If we have research notes but they need analysis, claims extraction, or structuring.\n"
                "- 'writer': If we have analyzed findings and need to write the final draft or refine it based on critic feedback.\n"
                "- 'critic': If we have a draft of the final answer and need to review it for citations, hallucination, or quality.\n"
                "- 'done': If the final answer is complete, accurate, and approved (or maximum iterations are close to limit).\n\n"
                "Return exactly a JSON object of this structure:\n"
                "{\n"
                "  \"next_step\": \"researcher\" | \"analyst\" | \"writer\" | \"critic\" | \"done\",\n"
                "  \"reason\": \"Explain why this routing step was chosen.\"\n"
                "}\n"
                "Do not output markdown codeblocks, only raw JSON."
            )

            state_summary = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Iteration: {state.iteration} / {settings.max_iterations}\n"
                f"Route History: {state.route_history}\n"
                f"Has Research Notes: {has_research}\n"
                f"Has Analysis Notes: {has_analysis}\n"
                f"Has Draft Answer: {has_final}\n"
                f"Latest Critic Feedback: {critic_feedback or 'None'}\n"
            )

            response = llm.complete(system_prompt, state_summary)
            content = response.content.strip()

            # Clean markdown code blocks if any
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            data = json.loads(content)
            next_step = data.get("next_step", fallback_step).lower()
            reason = data.get("reason", "LLM decided next step")

            # Validate next_step is in choices
            valid_choices = ["researcher", "analyst", "writer", "critic", "done"]
            if next_step not in valid_choices:
                logger.warning(f"LLM returned invalid routing choice '{next_step}'. Using fallback '{fallback_step}'.")
                next_step = fallback_step
                reason = f"Fallback due to invalid LLM step '{next_step}'"

            # Record supervisor decision in state
            state.record_route(next_step)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=next_step,
                    metadata={
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "reason": reason,
                    },
                )
            )
            state.add_trace_event("supervisor_decision", {"next_step": next_step, "reason": reason})

        except Exception as e:
            logger.warning(f"Error in Supervisor routing LLM call: {e}. Using fallback '{fallback_step}'")
            state.record_route(fallback_step)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=fallback_step,
                    metadata={"reason": f"Fallback due to error: {e}"},
                )
            )
            state.add_trace_event("supervisor_decision", {"next_step": fallback_step, "reason": f"Error: {e}"})

        return state
