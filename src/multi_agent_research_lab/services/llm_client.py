"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured in settings.")
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError)),
        reraise=True,
    )
    def _call_api(self, system_prompt: str, user_prompt: str) -> openai.types.chat.ChatCompletion:
        return self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=float(self.settings.timeout_seconds),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        try:
            response = self._call_api(system_prompt, user_prompt)
            content = response.choices[0].message.content or ""
            
            usage = response.usage
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                # Estimate cost
                model = self.settings.openai_model
                if "gpt-4o-mini" in model:
                    input_rate = 0.15 / 1_000_000
                    output_rate = 0.60 / 1_000_000
                elif "gpt-4o" in model:
                    input_rate = 2.50 / 1_000_000
                    output_rate = 10.00 / 1_000_000
                else:
                    input_rate = 0.15 / 1_000_000
                    output_rate = 0.60 / 1_000_000
                    
                cost_usd = (input_tokens * input_rate) + (output_tokens * output_rate)
            else:
                input_tokens = None
                output_tokens = None
                cost_usd = None
                
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        except Exception as e:
            logger.exception("LLM call failed")
            raise AgentExecutionError(f"LLM call failed: {e}") from e
