from agent_eval_framework.adapters.base import BaseAgentAdapter
from typing import Dict, Any

class DummyAgentAdapter(BaseAgentAdapter):
    def load_agent(self, **kwargs):
        return "dummy_agent"

    def get_response(self, prompt: str) -> Dict[str, Any]:
        return {
            "actual_response": f"response to {prompt}",
            "predicted_trajectory": []
        }
