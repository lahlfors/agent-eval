# Personalized Shopping Agent

## Overview of the Agent
This agent sample efficiently provides tailored product recommendations within the ecosystem of a specific brand, merchant, or online marketplace. It enhances the shopping experience within its own context by using targeted data and offering relevant suggestions.

## Agent Details
The personalized-shopping agent has the following capabilities:

*   Navigates specific websites to gather product information and understand available options.
*   Identifies suitable products using text and image-based search applied to a specific catalog.
*   Compares product features or within a defined brand or marketplace scope.
*   Recommends products based on user behavior and profile data.

The agent’s default configuration allows you to simulate interactions with a focused shopper. It demonstrates how an agent navigates a specific retail environment.

| Feature          | Description                                           |
| ---------------- | ----------------------------------------------------- |
| Interaction Type | Conversational                                        |
| Complexity       | Easy                                                  |
| Agent Type       | Single Agent                                          |
| Components       | Web Environment, SearchTool, ClickTool, Memory        |
| Vertical         | E-Commerce                                            |

## Setup and Installation
**Prerequisites:**
To begin, please firstly clone this repo, then install the required packages with the `poetry` command below.

```bash
# Navigate to the project root
cd personalized-shopping

# Install all dependencies for the agent and the evaluation framework
poetry install
```

**Data Setup:**
To run the agent, you must first download the JSON files containing the product information.

```bash
cd personalized_shopping/shared_libraries
mkdir data
cd data

# Download required data files using gdown
gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib;
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu;
# ... (and other data files as needed) ...
```
Then you need to index the product data so that they can be used by the search engine:
```bash
# From personalized_shopping/shared_libraries/
cd ../search_engine
mkdir -p resources_100 resources_1k resources_10k resources_50k
python convert_product_file_format.py

# Index the products
mkdir -p indexes
bash run_indexing.sh
cd ../../
```

## Running the Agent
You can interact with the agent via the command line or a web interface.

**Option 1: CLI**
```bash
adk run personalized_shopping
```

**Option 2: Web Interface**
```bash
cd personalized-shopping
adk web
```

## Evaluation Architecture
This project now features a modular, two-part evaluation system:

1.  **Agent Eval Framework:** A standalone, generic framework for running evaluations using the Vertex AI Evaluation Service. This framework is located in the `agent-eval-framework/` directory and is installed as a package. It knows how to run an evaluation but has no knowledge of any specific agent.
2.  **Personalized Shopping Agent:** This agent is now a *consumer* of the evaluation framework. It provides its own agent-specific logic (an "Adapter") and configuration to the framework.

This decoupled architecture allows the evaluation framework to be reused for any agent, while the agent itself only needs to provide a small amount of glue code to be evaluated.

## Running Evaluations
There are now two ways to evaluate the agent.

### 1. New: Vertex AI-based Evaluation (Recommended)
This is the new, primary method for evaluation. It uses the `agent-eval-framework` to run evaluations against a deployed agent using the powerful metrics from the Vertex AI Evaluation Service.

**Configuration:**

1.  **Deploy the Agent:** You must first deploy the agent to Vertex AI Agent Engine by following the "Deployment" instructions below.
2.  **Set Environment Variables:** Create a `.env` file in the project root and add the following, filling in your specific values:
    ```
    # .env file
    GCP_PROJECT_ID="your-gcp-project-id"
    GCP_REGION="your-gcp-region"
    AGENT_ENGINE_ID="your-deployed-agent-id"
    ```
3.  **Configure the Test:** The evaluation is configured in `eval/config.yaml`. You can change the dataset path or the metrics (`rouge`, `bleu`, etc.) in this file.

**Running the Test:**

Execute the following command from the project root:
```bash
poetry run pytest -s eval/test_vertex_eval.py
```
This will call your live agent, compare its responses to the golden dataset, and print a table of results.

### 2. Original ADK-based Evaluation
The original evaluation method provided by the ADK framework is still available. This is useful for testing local tool trajectories.

**Running the Test:**
```bash
poetry run pytest eval/test_eval.py
```

## Deployment
The personalized shopping agent sample can be deployed to Vertex AI Agent Engine.
```bash
# From the project root
cd deployment
python3 deploy.py
```
When the deployment finishes, it will print the `AGENT_ENGINE_ID` that you need for your `.env` file.
