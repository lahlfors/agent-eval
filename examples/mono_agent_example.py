# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example Mono-Agent (Single Agent) for Test Harness Demonstration."""

from typing import Any, Dict, List


class SampleCustomerSupportAgent:
    """Sample single agent with mock retrieval and calculator tools."""

    def __init__(self, model_name: str = "gemini-3.7-flash"):
        self.model_name = model_name

    def query(self, prompt: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        trajectory: List[Dict[str, Any]] = []

        if "return policy" in prompt_lower or "refund" in prompt_lower:
            trajectory.append({
                "tool_name": "search_knowledge_base",
                "tool_input": {"query": "return policy refund window"},
            })
            response = "Our return policy allows items to be returned within 30 days of purchase for a full refund."

        elif "shipping" in prompt_lower or "delivery" in prompt_lower:
            trajectory.append({
                "tool_name": "lookup_shipping_rates",
                "tool_input": {"country": "US", "method": "standard"},
            })
            response = "Standard shipping takes 3-5 business days and is free for orders over $50."

        elif "order" in prompt_lower and "status" in prompt_lower:
            trajectory.append({
                "tool_name": "get_order_status",
                "tool_input": {"order_id": "ORD-12345"},
            })
            response = "Your order ORD-12345 has shipped and is scheduled for delivery tomorrow."

        else:
            response = f"I am a customer support agent powered by {self.model_name}. How can I assist you today?"

        return {
            "response": response,
            "trajectory": trajectory,
            "metadata": {"model": self.model_name},
        }


# Singleton entrypoint
support_agent = SampleCustomerSupportAgent()


def run_agent(prompt: str) -> Dict[str, Any]:
    """Function entrypoint for CallableAgentAdapter."""
    return support_agent.query(prompt)
