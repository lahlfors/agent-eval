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

"""Unit tests for tokenomics and cost evaluation metrics."""

import pytest
from agent_eval_framework.metrics.tokenomics_metrics import (
    estimate_token_count,
    compute_cost_usd,
    calculate_comparative_roi,
    cost_savings_multiplier,
    token_cost_usd,
    multi_turn_token_growth_rate,
)


def test_estimate_token_count():
    assert estimate_token_count("") == 0
    assert estimate_token_count(None) == 0
    assert estimate_token_count("Hello world") > 0
    # Test structured payload
    payload = [{"tool_name": "search", "tool_input": {"query": "running shoes"}}]
    assert estimate_token_count(payload) > 10


def test_compute_cost_usd():
    # 1M input tokens on Gemini 3.7 Flash is $0.075
    cost_input_1m = compute_cost_usd(1_000_000, 0, model="gemini-3.7-flash")
    assert cost_input_1m == 0.075

    # 1M output tokens on Gemini 3.7 Flash is $0.30
    cost_output_1m = compute_cost_usd(0, 1_000_000, model="gemini-3.7-flash")
    assert cost_output_1m == 0.30

    # Claude 3.7 Sonnet 1M input is $3.00, 1M output is $15.00
    cost_claude = compute_cost_usd(1_000_000, 1_000_000, model="claude-3-7-sonnet")
    assert cost_claude == 18.0


def test_calculate_comparative_roi():
    roi = calculate_comparative_roi(
        input_tokens=1000,
        output_tokens=500,
        candidate_model="gemini-3.7-flash",
        baseline_model="claude-3.7-sonnet",
    )
    assert roi["candidate_cost_usd"] < roi["baseline_cost_usd"]
    assert roi["cost_multiplier_advantage"] > 10.0
    assert roi["savings_percentage"] > 90.0


def test_custom_metrics_evaltask_functions():
    prompt = "Find a pair of floral summer dresses under $50"
    reference = "Here are three options for floral summer dresses under $50..."
    actual_response = "I found a great floral summer dress for $42.99."
    actual_trajectory = [{"tool_name": "search", "tool_input": {"query": "floral summer dress"}}]

    # Test cost multiplier
    roi_score = cost_savings_multiplier(prompt, reference, actual_response, actual_trajectory)
    assert "score" in roi_score
    assert roi_score["score"] > 10.0
    assert "reason" in roi_score

    # Test token cost
    cost_score = token_cost_usd(prompt, reference, actual_response, actual_trajectory)
    assert "score" in cost_score
    assert cost_score["score"] > 0.0

    # Test trajectory growth
    growth_score = multi_turn_token_growth_rate(prompt, reference, actual_response, actual_trajectory)
    assert growth_score["score"] == 1.0
