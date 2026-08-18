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

"""Tokenomics and Cost Analysis Metrics for Agent Evaluation.

Provides metrics for token efficiency, USD execution cost calculations,
model-to-model cost comparisons (e.g., Gemini 3.7 Flash vs. Claude 3.7 Sonnet),
and multi-turn token growth trajectory analysis.
"""

import json
import math
from typing import Any, Dict, List, Optional, Union

# Model pricing catalog in USD per 1 Million tokens (Prompt / Completion)
MODEL_PRICING_CATALOG: Dict[str, Dict[str, float]] = {
    # Google Gemini Models
    "gemini-3.7-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30, "cached_input_per_1m": 0.01875},
    "gemini-2.5-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30, "cached_input_per_1m": 0.01875},
    "gemini-1.5-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30, "cached_input_per_1m": 0.01875},
    "gemini-2.5-pro": {"input_per_1m": 1.25, "output_per_1m": 5.00, "cached_input_per_1m": 0.3125},
    "gemini-1.5-pro": {"input_per_1m": 1.25, "output_per_1m": 5.00, "cached_input_per_1m": 0.3125},
    # Anthropic Claude Models
    "claude-3-7-sonnet": {"input_per_1m": 3.00, "output_per_1m": 15.00, "cached_input_per_1m": 0.30},
    "claude-3-5-sonnet": {"input_per_1m": 3.00, "output_per_1m": 15.00, "cached_input_per_1m": 0.30},
    "claude-3-haiku": {"input_per_1m": 0.25, "output_per_1m": 1.25, "cached_input_per_1m": 0.025},
    # OpenAI Models
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00, "cached_input_per_1m": 1.25},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60, "cached_input_per_1m": 0.075},
}


def estimate_token_count(content: Any) -> int:
    """Estimates the number of tokens in text or structured tool trajectory payload.

    Args:
        content: String, dict, list, or primitive to count tokens for.

    Returns:
        Estimated integer token count (using standard ~4 chars per token rule).
    """
    if content is None:
        return 0
    if isinstance(content, (dict, list)):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = str(content)

    if not text:
        return 0

    # Approx 4 characters per token average across English and JSON/code structures
    return max(1, math.ceil(len(text) / 4.0))


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().replace(".", "-").replace("_", "-").strip()
    return normalized

def get_model_pricing(model_name: str) -> Dict[str, float]:
    normalized = _normalize_model_name(model_name)
    if normalized in MODEL_PRICING_CATALOG:
        return MODEL_PRICING_CATALOG[normalized]
    # Check partial prefix matches
    for key, pricing in MODEL_PRICING_CATALOG.items():
        if key in normalized or normalized in key:
            return pricing
    return MODEL_PRICING_CATALOG["gemini-3.7-flash"]

def compute_cost_usd(input_tokens: int, output_tokens: int, model: str = "gemini-3.7-flash") -> float:
    """Calculates USD cost for given token counts and model pricing.

    Args:
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.
        model: Model identifier in MODEL_PRICING_CATALOG.

    Returns:
        Cost in USD (rounded to 8 decimal places).
    """
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
    return round(input_cost + output_cost, 8)


def calculate_comparative_roi(
    input_tokens: int,
    output_tokens: int,
    candidate_model: str = "gemini-3.7-flash",
    baseline_model: str = "claude-3.7-sonnet",
) -> Dict[str, Any]:
    """Computes comparative cost savings and ROI multiplier between candidate and baseline models.

    Args:
        input_tokens: Total prompt tokens.
        output_tokens: Total generated completion tokens.
        candidate_model: Evaluated model (e.g. Gemini 3.7 Flash).
        baseline_model: Reference comparator model (e.g. Claude 3.7 Sonnet).

    Returns:
        Dictionary containing candidate cost, baseline cost, savings amount, and multiplier.
    """
    candidate_cost = compute_cost_usd(input_tokens, output_tokens, model=candidate_model)
    baseline_cost = compute_cost_usd(input_tokens, output_tokens, model=baseline_model)

    savings_usd = round(baseline_cost - candidate_cost, 8)
    cost_multiplier = round(baseline_cost / candidate_cost, 2) if candidate_cost > 0 else 1.0
    savings_percentage = round(((baseline_cost - candidate_cost) / baseline_cost) * 100, 2) if baseline_cost > 0 else 0.0

    return {
        "candidate_model": candidate_model,
        "candidate_cost_usd": candidate_cost,
        "baseline_model": baseline_model,
        "baseline_cost_usd": baseline_cost,
        "savings_usd": savings_usd,
        "cost_multiplier_advantage": cost_multiplier,
        "savings_percentage": savings_percentage,
    }


# ============================================================================
# Vertex AI Custom Metric Functions (Evaluated per record in EvalTask)
# ============================================================================

def cost_savings_multiplier(
    prompt: str,
    reference: str,
    actual_response: str,
    actual_trajectory: list,
) -> Dict[str, Any]:
    """Custom Metric: Computes the cost multiplier advantage of Gemini 3.7 Flash vs. Claude 3.7 Sonnet.

    Returns:
        Score dictionary with the cost multiplier (e.g., 20.0x cheaper) and calculation details.
    """
    input_tokens = estimate_token_count(prompt)
    tool_overhead_tokens = estimate_token_count(actual_trajectory)
    total_input_tokens = input_tokens + tool_overhead_tokens
    output_tokens = estimate_token_count(actual_response)

    roi = calculate_comparative_roi(
        input_tokens=total_input_tokens,
        output_tokens=output_tokens,
        candidate_model="gemini-3.7-flash",
        baseline_model="claude-3.7-sonnet",
    )

    score = roi["cost_multiplier_advantage"]
    reason = (
        f"Gemini 3.7 Flash cost: ${roi['candidate_cost_usd']:.6f} vs "
        f"Claude 3.7 Sonnet cost: ${roi['baseline_cost_usd']:.6f} "
        f"({score}x cheaper, {roi['savings_percentage']}% savings)"
    )

    return {"score": float(score), "reason": reason}


def token_cost_usd(
    prompt: str,
    reference: str,
    actual_response: str,
    actual_trajectory: list,
) -> Dict[str, Any]:
    """Custom Metric: Calculates exact USD execution cost using Gemini 3.7 Flash pricing."""
    input_tokens = estimate_token_count(prompt) + estimate_token_count(actual_trajectory)
    output_tokens = estimate_token_count(actual_response)
    cost = compute_cost_usd(input_tokens, output_tokens, model="gemini-3.7-flash")

    return {
        "score": float(cost),
        "reason": f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${cost:.6f}",
    }


def multi_turn_token_growth_rate(
    prompt: str,
    reference: str,
    actual_response: str,
    actual_trajectory: list,
) -> Dict[str, Any]:
    """Custom Metric: Analyzes tool trajectory expansion and token accumulation efficiency.

    Scores 1.0 for efficient linear token accumulation and lower for excessive quadratic payload bloat.
    """
    traj_len = len(actual_trajectory) if isinstance(actual_trajectory, list) else 0
    if traj_len == 0:
        return {"score": 1.0, "reason": "Single turn / no tool calls. Linear token growth."}

    traj_tokens = estimate_token_count(actual_trajectory)
    avg_tokens_per_step = traj_tokens / max(1, traj_len)

    # Penalize if average step overhead exceeds 1,500 tokens (sign of uncompacted HTML/JSON payload bloat)
    if avg_tokens_per_step <= 500:
        score = 1.0
        reason = f"Highly efficient trajectory ({traj_len} steps, {avg_tokens_per_step:.0f} tokens/step average)."
    elif avg_tokens_per_step <= 1500:
        score = 0.8
        reason = f"Moderate trajectory payload ({traj_len} steps, {avg_tokens_per_step:.0f} tokens/step average)."
    else:
        score = max(0.2, 1.0 - (avg_tokens_per_step / 5000.0))
        reason = f"Large payload expansion detected ({traj_len} steps, {avg_tokens_per_step:.0f} tokens/step average)."

    return {"score": round(score, 2), "reason": reason}
