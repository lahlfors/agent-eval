# /Users/laah/Code/walmart/agent-eval/eval/test_main_eval.py
import pytest
import os
from agent_eval_framework.runner import run_evaluation

# Optional: Load .env for local runs if not handled by shell
from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH = "agent-eval-framework/config/adk_eval_config.yaml"

def test_shopping_agent_vertex_eval():
    """
    Triggers the agent evaluation using the agent-eval-framework,
    which uses the Vertex AI GenAI Evaluation Service.
    """
    if not os.getenv("GCP_PROJECT_ID") or not os.getenv("GCP_REGION"):
        pytest.skip("GCP_PROJECT_ID and GCP_REGION must be set in .env file to run Vertex AI evaluations.")

    print(f"Running evaluation with config: {CONFIG_PATH}")
    try:
        eval_result = run_evaluation(config_path=CONFIG_PATH)
        assert eval_result is not None, "Evaluation failed to produce results."
        print("Evaluation completed successfully using agent-eval-framework.")
        # Add assertions based on eval_result.summary_metrics if desired
        # Example: assert eval_result.summary_metrics.get('exact_match', 0) > 0.6
    except Exception as e:
        pytest.fail(f"run_evaluation failed: {e}")
