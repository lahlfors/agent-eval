# Monorepo for Agent Development and Evaluation

This repository serves as a comprehensive example of a well-structured, modular system for developing, deploying, and evaluating AI agents. It contains multiple independent but related packages, showcasing best practices for code organization and separation of concerns.

## Repository Structure

This repository is a "monorepo" containing several distinct Python packages:

-   **`agent-eval-framework/`**: A powerful, standalone framework for evaluating agents using the Vertex AI Evaluation Service. It is designed to be generic and reusable for any agent.
-   **`well_structured_system/`**: A sample application demonstrating principles of good software architecture, including dependency injection, asynchronous programming, and clear separation of concerns (app logic vs. evaluation logic).
-   **`personalized_shopping/`**: A concrete example of a conversational AI agent built for e-commerce. This agent acts as a *consumer* of the `agent-eval-framework` to demonstrate how a real-world agent can be evaluated.
-   **`eval/`**: Contains configuration and test scripts for running evaluations on the `personalized_shopping` agent.

## Core Concepts

### 1. Decoupled Evaluation Framework (`agent-eval-framework`)

The cornerstone of this repository is the **`agent-eval-framework`**. Its key features are:

-   **Generic**: It has no knowledge of any specific agent's implementation details.
-   **Configuration-Driven**: All aspects of an evaluation (the agent to test, the dataset to use, the metrics to run) are defined in a single `config.yaml` file.
-   **Extensible**: It uses an "Adapter" pattern, allowing you to easily make any agent compatible with the framework by writing a simple wrapper class.
-   **Powered by Vertex AI**: It leverages the robust and scalable metrics of the Google Vertex AI Evaluation Service.

By installing this framework as a package, you can use it to evaluate any number of different agents without duplicating evaluation logic.

### 2. Modular Application (`well_structured_system`)

The `well_structured_system` is an ideal starting point for building robust applications. It demonstrates:

-   **Asynchronous Operations**: Using `asyncio` for efficient, non-blocking tool calls.
-   **Caching**: A simple, swappable caching mechanism to improve performance.
-   **Separation of Concerns**: Application code (`src/app`) is cleanly separated from its evaluation logic (`src/evaluation`).

### 3. Example Agent (`personalized_shopping`)

The `personalized_shopping` agent is a practical example that simulates a retail shopping assistant. It showcases how to:

-   Define an agent with tools (`search`, `click`).
-   Create a custom **Adapter** (`PersonalizedShoppingAgentAdapter`) to make the agent compatible with the evaluation framework.
-   Configure and run evaluations against a live, deployed version of the agent.

## Setup and Installation

**Prerequisites:**
-   Python 3.9+
-   [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install all dependencies:**
    This single command installs the dependencies for all packages in the monorepo, including the `agent-eval-framework` in an editable mode.
    ```bash
    poetry install
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the project root by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and fill in your specific Google Cloud project details:
    ```
    # .env
    GCP_PROJECT_ID="your-gcp-project-id"
    GCP_REGION="your-gcp-region"
    # This will be filled in after you deploy the agent
    AGENT_ENGINE_ID=""
    ```

## Running the Evaluation

The primary workflow demonstrated in this repository is evaluating the `personalized_shopping` agent using the `agent-eval-framework`.

### 1. Deploy the Agent

Before you can run the evaluation, you must deploy the `personalized_shopping` agent to Vertex AI Agent Engine.

```bash
# From the project root
cd deployment
python3 deploy.py
```

This script will deploy the agent and print its `AGENT_ENGINE_ID`. **Copy this ID and paste it into your `.env` file.**

### 2. Configure the Evaluation

The evaluation run is configured in `eval/config.yaml`. Here you can specify:
-   The **agent adapter** to use.
-   The **golden dataset** for the evaluation.
-   The **metrics** to compute (e.g., `rouge`, `bleu`, `trajectory_exact_match`).

### 3. Execute the Evaluation

Run the evaluation using `pytest`. This will trigger the `agent-eval-framework` to:
1.  Read the `eval/config.yaml`.
2.  Instantiate the `VertexAgentEngineAdapter`.
3.  Connect to your deployed agent using the `AGENT_ENGINE_ID` from your `.env` file.
4.  Run through the specified dataset, comparing your agent's live responses to the golden reference.
5.  Print a table of results.

```bash
# From the project root
poetry run pytest -s eval/test_vertex_eval.py
```

## Running the Demos

This repository also includes demos for the `well_structured_system`.

-   **Run the App Demo:** See the asynchronous application logic in action.
    ```bash
    poetry run python well_structured_system/run_app_demo.py
    ```
-   **Run the Evaluation Demo:** See the local evaluation pipeline for the `well_structured_system` in action.
    ```bash
    poetry run python well_structured_system/run_evaluation_demo.py
    ```
