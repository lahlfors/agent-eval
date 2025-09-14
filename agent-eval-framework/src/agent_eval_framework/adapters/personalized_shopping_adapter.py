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
            async def _get_response():
                response = ""
                # The ADK agent's root_agent has a `process` method which is an async generator.
                async for r in self.agent.process([prompt]):
                    response = r
                return response

            final_response = asyncio.run(_get_response())

            log.info("Adapter received response", extra={"response": final_response})
            return {"actual_response": final_response}
        except Exception as e:
            log.error("Error during agent interaction", exc_info=True)
            return {"actual_response": "AGENT_EXECUTION_ERROR"}
