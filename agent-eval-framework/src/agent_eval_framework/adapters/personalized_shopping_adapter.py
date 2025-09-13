import importlib
from typing import Dict, Any
from agent_eval_framework.adapters.base import BaseAgentAdapter
from agent_eval_framework.utils.logger import get_logger
import asyncio

log = get_logger(__name__)

class PersonalizedShoppingAdapter(BaseAgentAdapter):
    """Adapter for the Personalized Shopping ADK agent."""

    def load_agent(self, agent_module_path: str = "personalized_shopping.agent"):
        try:
            module = importlib.import_module(agent_module_path)
            if hasattr(module, "root_agent"):
                log.info(f"Successfully loaded root_agent from {agent_module_path}")
                return module.root_agent
            else:
                raise ImportError(f"root_agent not found in {agent_module_path}")
        except Exception as e:
            log.error(f"Failed to load agent from {agent_module_path}", exc_info=True)
            raise

    def get_response(self, prompt: str) -> Dict[str, Any]:
        if not self.agent:
            return {"actual_response": "AGENT_NOT_LOADED"}

        log.info("Adapter getting response", extra={"prompt": prompt})
        try:
            # The ADK agent's root_agent has a `process` method that takes a list of strings
            # and returns a list of strings. We use asyncio.run to execute it.
            response_generator = self.agent.process([prompt])

            # The response is a generator; we need to extract the string content.
            # We'll take the first response.
            response = ""
            for r in response_generator:
                response = r
                break # Assuming we only want the first response

            log.info("Adapter received response", extra={"response": response})
            return {"actual_response": response}
        except Exception as e:
            log.error("Error during agent interaction", exc_info=True)
            return {"actual_response": "AGENT_EXECUTION_ERROR"}
