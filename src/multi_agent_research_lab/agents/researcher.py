"""Researcher agent skeleton."""

import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("ResearcherAgent executing search...")
        
        llm = LLMClient()
        search_client = SearchClient()
        
        total_in_tokens = 0
        total_out_tokens = 0
        total_cost = 0.0

        # Step 1: Formulate search query using LLM
        try:
            query_sys_prompt = (
                "You are the Researcher Agent of a Multi-Agent Research System.\n"
                "Given the user's research request, generate a single optimized search engine query to find "
                "the most relevant, up-to-date information. Return ONLY the raw query string. Do not use quotes or markdown."
            )
            query_user_prompt = f"User Request: {state.request.query}"
            query_resp = llm.complete(query_sys_prompt, query_user_prompt)
            
            search_query = query_resp.content.strip().strip('"').strip("'")
            if query_resp.input_tokens:
                total_in_tokens += query_resp.input_tokens
            if query_resp.output_tokens:
                total_out_tokens += query_resp.output_tokens
            if query_resp.cost_usd:
                total_cost += query_resp.cost_usd
        except Exception as e:
            logger.warning(f"Failed to formulate search query using LLM: {e}. Using original query.")
            search_query = state.request.query

        # Step 2: Query the SearchClient
        try:
            logger.info(f"Formulated search query: '{search_query}'")
            new_sources = search_client.search(search_query, max_results=state.request.max_sources)
            
            # Avoid duplicate sources in state.sources
            existing_urls = {src.url for src in state.sources if src.url}
            for src in new_sources:
                if not src.url or src.url not in existing_urls:
                    state.sources.append(src)
        except Exception as e:
            logger.error(f"Search failed in ResearcherAgent: {e}")
            state.errors.append(f"ResearcherAgent search error: {e}")

        # Step 3: Compile research notes
        notes_sys_prompt = (
            "You are the Researcher Agent of a Multi-Agent Research System.\n"
            "Your task is to review all gathered search sources and compile detailed, factual research notes.\n"
            "Group notes into clear logical sections. For every key fact, explicitly cite the title/URL of the source.\n"
            "If no sources are found or the search was empty, indicate that clearly and summarize what is known."
        )
        
        sources_text = ""
        for i, src in enumerate(state.sources):
            sources_text += f"[{i+1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"
            
        notes_user_prompt = (
            f"User Research Query: {state.request.query}\n"
            f"Gathered Sources:\n{sources_text or 'No sources available.'}"
        )
        
        try:
            notes_resp = llm.complete(notes_sys_prompt, notes_user_prompt)
            state.research_notes = notes_resp.content
            
            if notes_resp.input_tokens:
                total_in_tokens += notes_resp.input_tokens
            if notes_resp.output_tokens:
                total_out_tokens += notes_resp.output_tokens
            if notes_resp.cost_usd:
                total_cost += notes_resp.cost_usd
        except Exception as e:
            logger.exception("Failed to compile research notes")
            state.errors.append(f"ResearcherAgent notes compiler error: {e}")
            state.research_notes = f"Failed to compile research notes due to: {e}"

        # Record AgentResult
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata={
                    "tokens_in": total_in_tokens,
                    "tokens_out": total_out_tokens,
                    "cost_usd": total_cost,
                    "search_query": search_query,
                    "sources_found": len(state.sources),
                },
            )
        )
        state.add_trace_event("research_completed", {"sources_count": len(state.sources)})
        return state
