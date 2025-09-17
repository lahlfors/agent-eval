from agent_eval_framework.runner import run_evaluation
import os

if __name__ == "__main__":
    """
    Runner script for the agent evaluation framework.
    """
    # The config file is located in the 'config' directory relative to this script.
    config_path = os.path.join(os.path.dirname(__file__), "config", "adk_eval_config.yaml")

    print(f"Starting evaluation with config file: {config_path}")

    # Ensure the config file exists before running
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
    else:
        run_evaluation(config_path=config_path)
