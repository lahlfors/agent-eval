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

"""Unit tests for Mono-Agent and Multi-Agent Harness Adapters."""

import pytest
from unittest.mock import MagicMock
from agent_eval_framework.adapters import (
    BaseAgentAdapter,
    CallableAgentAdapter,
    MultiAgentSystemAdapter,
    HttpAgentAdapter,
    normalize_adapter_output,
)


def test_normalize_adapter_output():
    out = normalize_adapter_output(
        response_text="Test response",
        trajectory=[{"tool_name": "search", "tool_input": {}}],
        multi_agent_events=[{"author": "AgentA", "content": "Delegated"}],
    )
    assert out["response"] == "Test response"
    assert out["actual_response"] == "Test response"
    assert len(out["predicted_trajectory"]) == 1
    assert len(out["multi_agent_events"]) == 1


def test_callable_agent_adapter_with_function():
    def dummy_agent(prompt: str):
        return {
            "response": f"Echo: {prompt}",
            "trajectory": [{"tool_name": "echo_tool", "tool_input": {"text": prompt}}],
        }

    adapter = CallableAgentAdapter(target=dummy_agent)
    res = adapter.get_response("Hello test harness")
    assert "Echo: Hello test harness" in res["response"]
    assert len(res["predicted_trajectory"]) == 1
    assert res["predicted_trajectory"][0]["tool_name"] == "echo_tool"


def test_callable_agent_adapter_with_dotpath():
    adapter = CallableAgentAdapter(target="examples.mono_agent_example:run_agent")
    res = adapter.get_response("What is the return policy?")
    assert "30 days" in res["response"]
    assert len(res["predicted_trajectory"]) == 1
    assert res["predicted_trajectory"][0]["tool_name"] == "search_knowledge_base"


def test_multi_agent_system_adapter():
    adapter = MultiAgentSystemAdapter(
        coordinator_entrypoint="examples.multi_agent_example:run_multi_agent_system",
        topology="hierarchical",
        sub_agents=["ResearchSpecialist", "CalculationSpecialist"],
    )

    # Test Math / Calculation delegation
    calc_res = adapter.get_response("Calculate ROI for project")
    assert "Calculation Complete" in calc_res["response"]
    assert len(calc_res["multi_agent_events"]) >= 2
    assert "CalculationSpecialist" in calc_res["metadata"]["participating_agents"]

    # Test Research delegation
    research_res = adapter.get_response("Research latest market trends")
    assert "Research Synthesis" in research_res["response"]
    assert "ResearchSpecialist" in research_res["metadata"]["participating_agents"]


def test_http_agent_adapter(mocker):
    mock_post = mocker.patch("requests.post")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Remote agent response",
        "trajectory": [{"tool_name": "remote_db_lookup", "tool_input": {"id": 1}}],
        "multi_agent_events": [{"author": "RemoteWorker", "action": "done"}],
    }
    mock_post.return_value = mock_resp

    adapter = HttpAgentAdapter(endpoint_url="https://api.example.com/agent")
    res = adapter.get_response("Test remote query")

    assert res["response"] == "Remote agent response"
    assert len(res["predicted_trajectory"]) == 1
    assert res["metadata"]["status_code"] == 200
