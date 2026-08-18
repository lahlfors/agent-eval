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

"""Example Multi-Agent System (MAS) for Test Harness Demonstration.

Demonstrates a hierarchical multi-agent architecture with:
1. Coordinator / Supervisor Agent (Intent routing & delegation)
2. Research Specialist Sub-Agent (Information retrieval)
3. Calculation Specialist Sub-Agent (Formula & financial calculations)
4. Summarizer Sub-Agent (Formatting & synthesis)
"""

from typing import Any, Dict, List


class ResearchSubAgent:
    def execute(self, query: str) -> Dict[str, Any]:
        return {
            "agent": "ResearchSpecialist",
            "findings": f"Research retrieved 3 articles regarding '{query}'.",
            "tool_calls": [{"tool_name": "web_search", "tool_input": {"query": query}}],
        }


class CalculationSubAgent:
    def execute(self, expression: str) -> Dict[str, Any]:
        return {
            "agent": "CalculationSpecialist",
            "result": "Calculated ROI is +145% over 12 months.",
            "tool_calls": [{"tool_name": "run_python_calc", "tool_input": {"code": "compute_roi()"}}],
        }


class MultiAgentCoordinator:
    """Coordinator Agent routing requests to specialist sub-agents."""

    def __init__(self):
        self.researcher = ResearchSubAgent()
        self.calculator = CalculationSubAgent()

    def run_mas(self, prompt: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        events: List[Dict[str, Any]] = []
        consolidated_trajectory: List[Dict[str, Any]] = []
        participating_agents: List[str] = ["CoordinatorAgent"]

        # Routing decision logic
        if "calculate" in prompt_lower or "roi" in prompt_lower or "math" in prompt_lower:
            events.append({
                "author": "CoordinatorAgent",
                "action": "delegate_task",
                "target_subagent": "CalculationSpecialist",
                "instruction": f"Compute metrics for query: {prompt}",
            })
            calc_out = self.calculator.execute(prompt)
            participating_agents.append("CalculationSpecialist")
            events.append({
                "author": "CalculationSpecialist",
                "response": calc_out["result"],
            })
            consolidated_trajectory.extend(calc_out["tool_calls"])
            final_response = f"Calculation Complete: {calc_out['result']}"

        elif "research" in prompt_lower or "search" in prompt_lower or "market" in prompt_lower:
            events.append({
                "author": "CoordinatorAgent",
                "action": "delegate_task",
                "target_subagent": "ResearchSpecialist",
                "instruction": f"Conduct research on: {prompt}",
            })
            research_out = self.researcher.execute(prompt)
            participating_agents.append("ResearchSpecialist")
            events.append({
                "author": "ResearchSpecialist",
                "response": research_out["findings"],
            })
            consolidated_trajectory.extend(research_out["tool_calls"])
            final_response = f"Research Synthesis: {research_out['findings']}"

        else:
            final_response = "Coordinator: General query resolved directly without sub-agent delegation."

        return {
            "response": final_response,
            "trajectory": consolidated_trajectory,
            "multi_agent_events": events,
            "participating_agents": participating_agents,
            "turns": len(events) + 1,
            "token_usage": {"input_tokens": 420, "output_tokens": 180},
        }


# Singleton entrypoint
mas_coordinator = MultiAgentCoordinator()


def run_multi_agent_system(prompt: str) -> Dict[str, Any]:
    """Function entrypoint for MultiAgentSystemAdapter."""
    return mas_coordinator.run_mas(prompt)
