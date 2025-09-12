# Demonstration Guide: Local Agent with Cloud Evaluation

This guide provides a step-by-step walkthrough for demonstrating the key capabilities of this repository, focusing on the hybrid model of a locally running agent evaluated by a cloud-based service.

**Prerequisite:** You must complete all the steps in the `DEPLOYMENT.md` guide before proceeding.

## Part 1: Running the Agent Interactively (Local)

This demonstrates the agent's core conversational and tool-use capabilities in a simple, local, interactive loop.

1.  **Create an Interactive Script:**
    Create a new file in the project root named `run_local_demo.py` and paste the following code into it:

    ```python
    # run_local_demo.py
    from personalized_shopping.agent import agent
    from rich import print

    print("[bold green]Personalized Shopping Agent Demo[/bold green]")
    print("Type 'exit' to end the session.")
    print("-" * 50)

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                print("[bold yellow]Session ended.[/bold yellow]")
                break

            print(f"[cyan]Agent is thinking...[/cyan]")
            response = agent.query(user_input)
            print(f"[bold magenta]Agent:[/bold magenta] {response['output']}")
        except KeyboardInterrupt:
            print("\n[bold yellow]Session ended.[/bold yellow]")
            break
        except Exception as e:
            print(f"[bold red]An error occurred: {e}[/bold red]")

    ```

2.  **Run the Demo:**
    Execute the script from your terminal using Poetry.
    ```bash
    poetry run python run_local_demo.py
    ```
    You can now chat with the agent. It will use the local data and search index you configured.

## Part 2: Running the Hybrid Evaluation

This is the core demonstration. It shows how the locally running agent can be evaluated using the `agent-eval-framework`, which sends the agent's responses to the Vertex AI Evaluation Service in the cloud.

1.  **Ensure you are logged into gcloud:**
    The evaluation framework needs to authenticate with your Google Cloud account.
    ```bash
    gcloud auth application-default login
    ```

2.  **Execute the Evaluation Script:**
    Run the evaluation using `pytest`. The framework will automatically use the `LocalAgentAdapter` as configured in `eval/config.yaml`.

    ```bash
    # From the project root
    poetry run pytest -s eval/test_vertex_eval.py
    ```

3.  **Interpreting the Output:**
    The script will:
    *   Load the `LocalAgentAdapter`.
    *   Instantiate the `personalized_shopping` agent in memory.
    *   Run through the evaluation dataset (`eval/vertex_eval_data/golden_record.jsonl`).
    *   For each example, it will call the local agent's `query` method.
    *   It will send the agent's responses and the reference responses to the Vertex AI API.
    *   Finally, it will print a table of evaluation results (e.g., `rouge`, `trajectory_exact_match`).

## Part 3: Understanding the `well_structured_system`

The `well_structured_system/` directory is a self-contained, conceptual example of good software architecture. It is **not** functionally connected to the `personalized_shopping` agent, but serves as a blueprint for best practices.

*   **Separation of Concerns:** Note the clean separation between application logic (`src/app`) and evaluation logic (`src/evaluation`).
*   **Local Evaluation Example:** This system has its own local evaluation pipeline (`run_evaluation_demo.py`) and a `LibraryAdapter` (`src/evaluation/adapters.py`) which served as the inspiration for the `LocalAgentAdapter` used in this demo.

You can run its demos to see these principles in action:
```bash
# Run the app logic demo
poetry run python well_structured_system/run_app_demo.py

# Run the self-contained local evaluation demo
poetry run python well_structured_system/run_evaluation_demo.py
```

## Part 4: Code Modifications Report

To enable this local agent/cloud evaluation workflow, the following modifications were made:

1.  **Created `personalized_shopping/local_agent_adapter.py`:**
    A new `LocalAgentAdapter` class was created. This adapter imports the agent object directly and calls its `query` method, allowing the evaluation framework to run the agent without needing it to be deployed on a server.

2.  **Modified `eval/config.yaml`:**
    The configuration was updated to point to the new `LocalAgentAdapter`. The `agent_adapter_class` was changed, and the `agent_config` (which specified a deployment ID) was removed.

These changes decouple the agent's execution environment from the evaluation framework, providing greater flexibility for development and testing.
