# Copyright 2025 Google LLC
# ... (license headers) ...

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
            # This is a synchronous wrapper for the async ADK agent interaction
            return asyncio.run(self.get_response_async(prompt))
        except Exception as e:
            log.error("Error during agent interaction", exc_info=True)
            return {"actual_response": "AGENT_EXECUTION_ERROR"}

    async def get_response_async(self, prompt: str) -> Dict[str, Any]:
         from google.adk.apps import App
         my_app = App(name="personalized_shopping_app", root_agent=self.agent)
         async with my_app.create_session() as session:
            response_text = ""
            async for event in session.send_message(prompt):
                if event.content and event.content.parts:
                    response_text += event.content.parts[0].text
            log.info("Adapter received async response", extra={"response": response_text})
            return {"actual_response": response_text}
