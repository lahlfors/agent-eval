import os

import dotenv
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_tools():
    """Test the agent's basic ability on a few examples."""
    await AgentEvaluator.evaluate(
        "personalized_shopping",
        os.path.join(os.path.dirname(__file__), "tools"),
        num_runs=1,
    )
