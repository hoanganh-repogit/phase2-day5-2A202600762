"""Search client abstraction for ResearcherAgent."""

import json
import logging
import requests

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient


logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        If Tavily API key is available, queries the Tavily API.
        Otherwise, falls back to simulating search results using the LLM.
        """
        api_key = self.settings.tavily_api_key
        if api_key:
            try:
                logger.info(f"Querying Tavily for query: '{query}'")
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                    timeout=15.0,
                )
                response.raise_for_status()
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append(
                        SourceDocument(
                            title=item.get("title", "Untitled Source"),
                            url=item.get("url"),
                            snippet=item.get("content", ""),
                        )
                    )
                return results
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to LLM search simulation.")

        # Fallback Search Simulation using LLM
        logger.info(f"Using LLM fallback simulation for query: '{query}'")
        try:
            llm = LLMClient()
            system_prompt = (
                "You are an expert search engine simulation. Given a query, return a JSON list of search results. "
                "The output must be a valid JSON array of objects. Each object must have these exact fields:\n"
                "- 'title': The title of the web page/source document.\n"
                "- 'url': A realistic fake URL for the source (e.g. 'https://arxiv.org/abs/...').\n"
                "- 'snippet': A rich, detailed summary snippet containing facts, numbers, and technical details relevant to the query.\n\n"
                "Provide highly informative snippets based on actual SOTA knowledge. Do not include markdown codeblocks, only return the raw JSON array."
            )
            user_prompt = f"Query: {query}\nMax results: {max_results}"
            response = llm.complete(system_prompt, user_prompt)
            content = response.content.strip()
            
            # Clean markdown code blocks if any
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            items = json.loads(content)
            results = []
            for item in items[:max_results]:
                results.append(
                    SourceDocument(
                        title=item.get("title", "Simulated Source"),
                        url=item.get("url"),
                        snippet=item.get("snippet", ""),
                    )
                )
            return results
        except Exception as e:
            logger.exception("LLM search simulation failed. Returning basic static mock.")
            # Final fallback
            return [
                SourceDocument(
                    title=f"Static Mock Source - {query}",
                    url="https://example.com/mock-search",
                    snippet=f"This is a mock search snippet for query '{query}'. No Tavily key or LLM connection was successful.",
                )
            ]
