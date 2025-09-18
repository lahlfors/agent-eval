# Personalized Shopping Agent & Evaluation Framework

This repository serves as a comprehensive example of a well-structured, modular system for developing, deploying, and evaluating AI agents. It contains a powerful, reusable evaluation framework and a concrete example of a personalized shopping agent that uses it.

## Repository Structure

This repository is a "monorepo" containing several distinct Python packages:

-   **`agent-eval-framework/`**: A powerful, standalone framework for evaluating agents using the Vertex AI Evaluation Service. It is designed to be generic and reusable for any agent.
-   **`personalized_shopping/`**: A concrete example of a conversational AI agent built for e-commerce. This agent acts as a *consumer* of the `agent-eval-framework` to demonstrate how a real-world agent can be evaluated.
-   **`deployment/`**: Contains scripts for deploying the `personalized_shopping` agent to Vertex AI Agent Engine.
-   **`tests/`**: Contains unit tests for the `personalized_shopping` agent's tools.

## Core Concepts

### Decoupled Evaluation Framework (`agent-eval-framework`)

The cornerstone of this repository is the **`agent-eval-framework`**. Its key features are:

-   **Generic**: It has no knowledge of any specific agent's implementation details.
-   **Configuration-Driven**: All aspects of an evaluation are defined in configuration files.
-   **Extensible**: It uses an "Adapter" pattern, allowing you to easily make any agent compatible with the framework by writing a simple wrapper class.
-   **Powered by Vertex AI**: It leverages the robust and scalable metrics of the Google Vertex AI Evaluation Service.

By installing this framework as a package, you can use it to evaluate any number of different agents without duplicating evaluation logic.

---

## The Personalized Shopping Agent

### Overview

This agent sample efficiently provides tailored product recommendations within the ecosystem of a specific brand, merchant, or online marketplace. It enhances the shopping experience within its own context by using targeted data and offering relevant suggestions.

The personalized-shopping agent has the following capabilities:

*   Navigates specific websites to gather product information and understand available options.
*   Identifies suitable products using text and image-based search applied to a specific catalog.
*   Compares product features or within a defined brand or marketplace scope.
*   Recommends products based on user behavior and profile data.

| <div align="center">Feature</div> | <div align="center">Description</div> |
| --- | --- |
| <div align="center">**Interaction Type**</div> | <div align="center">Conversational</div> |
| <div align="center">**Complexity**</div>  | <div align="center">Easy</div> |
| <div align="center">**Agent Type**</div>  | <div align="center">Single Agent</div> |
| <div align="center">**Components**</div>  | <div align="center">Web Environment: Access to a pre-indexed product website</div> |
|  | <div align="center">SearchTool (to retrieve relevant product information)</div> |
|  | <div align="center">ClickTool (to navigate the website)</div> |
|  | <div align="center">Conversational Memory</div> |
| <div align="center">**Vertical**</div>  | <div align="center">E-Commerce</div> |

### Architecture
![Personalized Shopping Agent Architecture](ps_architecture.png)

### Key Features

The key features of the personalized-shopping agent include:
*  **Environment:** The agent can interact in an e-commerce web environment with 1.18M of products.
*  **Memory:** The agent maintains a conversational memory with all the previous-turns information in its context window.
*  **Tools:**
    *   _Search_: The agent has access to a search-retrieval engine, where it can perform key-word search for the related products.
    *   _Click_: The agent has access to the product website and it can navigate the website by clicking buttons.

---

## Setup and Installation

**Prerequisites:**
*   You should be using [poetry](https://python-poetry.org/docs/) for dependency management.
*   Your repo should contain a `.env.example` file highlighting what environmental variables are used.

1.  **Clone and Setup Python Environment:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    python3 -m venv myenv
    source myenv/bin/activate
    pip install poetry
    ```

2.  **Install Dependencies:**
    This single command installs the dependencies for all packages in the monorepo.
    ```bash
    # Note for Linux users: If you get an error related to `keyring` during the installation, you can disable it by running:
    # poetry config keyring.enabled false
    poetry install
    ```

3.  **Download Product Data:**
    Download the JSON files containing the product information, which are necessary to initialize the web environment.
    ```bash
    cd personalized_shopping/shared_libraries
    mkdir data
    cd data

    # Download items_shuffle_1000 (4.5MB)
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib

    # Download items_ins_v2_1000 (147KB)
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu

    # Download items_shuffle (5.1GB)
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB

    # Download items_ins_v2 (178MB)
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi

    # Download items_human_ins (4.9MB)
    gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
    ```

4.  **Index Product Data:**
    Convert and index the product data for the search engine.
    ```bash
    cd ../search_engine
    mkdir -p resources_100 resources_1k resources_10k resources_50k
    python convert_product_file_format.py

    mkdir -p indexes
    bash run_indexing.sh
    cd ../../../ # Return to project root
    ```

5.  **Configure Environment Variables:**
    Update the `.env.example` file with your cloud project name and region, then rename it to `.env`.
    ```bash
    cp .env.example .env
    # Now edit .env with your details
    ```

6.  **Authenticate Google Cloud:**
    ```bash
    gcloud auth application-default login
    ```

## Running the Agent

> **Note**: The first run may take some time as the system loads approximately 50,000 product entries into the web environment for the search engine.

- **Option 1: CLI**
    Talk to the agent using the command line.
    ```bash
    adk run personalized_shopping
    ```

- **Option 2: Web Interface**
    Run the agent on a web interface.
    ```bash
    adk web
    ```
    Select the `personalized_shopping` option from the dropdown list.

### Example Interactions

*   **Text-based search**: See [text_search_floral_dress.session.md](tests/example_interactions/text_search_floral_dress.session.md)
*   **Image-based search**: See [image_search_denim_skirt.session.md](tests/example_interactions/image_search_denim_skirt.session.md) (Input image: [example_product.png](tests/example_interactions/example_product.png))

---

## Running Evaluation and Tests

The evaluation assesses the agent's performance on tasks defined in the evalset.

### Run Agent Evaluation

The evaluation of the agent can be run from the project root directory using the `pytest` module:
```bash
python3 -m pytest eval
```
You can add more eval prompts by adding your dataset into the `eval/eval_data` folder. The judgment criteria are specified in `test_config.json`.

### Run Tool Unit Tests

To run unit tests for the agent's tools, run the following command from the project root directory:
```bash
python3 -m pytest tests
```

---

## Deployment

The personalized shopping agent can be deployed to Vertex AI Agent Engine.

1.  **Build the Wheel File**
    From the `personalized_shopping` directory:
    ```bash
    cd personalized_shopping
    poetry build --format=wheel --output=deployment
    cd .. # Return to project root
    ```

2.  **Deploy to Agent Engine**
    From the project root:
    ```bash
    cd deployment
    python3 deploy.py
    cd .. # Return to project root
    ```
    This process can take over 10 minutes. When it finishes, it will print the `AGENT_ENGINE_ID`.

3.  **Interact with Deployed Agent**
    You can interact with the deployed agent programmatically in Python:
    ```python
    import dotenv
    dotenv.load_dotenv()
    from vertexai import agent_engines

    agent_engine_id = "YOUR_AGENT_ENGINE_ID" # Remember to update the ID here.
    user_input = "Hello, can you help me find a summer dress? I want something flowy and floral."

    agent_engine = agent_engines.get(agent_engine_id)
    session = agent_engine.create_session(user_id="new_user")
    for event in agent_engine.stream_query(
        user_id=session["user_id"], session_id=session["id"], message=user_input
    ):
        for part in event["content"]["parts"]:
            print(part["text"])
    ```

4.  **Delete Deployed Agent**
    ```bash
    python3 deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
    ```

## Customization

This agent sample uses the webshop environment from [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop), which includes 1.18 million real-world products.

By default, the agent loads 50,000 products. You can adjust this by modifying the `num_product_items` parameter in `personalized_shopping/shared_libraries/init_env.py`.

You can add your own product data and place the annotations in `items_human_ins.json`, `items_ins_v2.json`, and `items_shuffle.json`.

## Troubleshooting

*   **Q:** I'm having issues with `gdown` during agent setup. What should I do?
*   **A:** You can manually download the files from the individual Google Drive links and place them in the `personalized_shopping/personalized_shopping/shared_libraries/data` folder.

## Acknowledgement
We are grateful to the developers of [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) for their simulated environment. This agent incorporates modified code from their project.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents. This sample has not been rigorously tested and may contain bugs or limitations. Users are solely responsible for any further development, testing, and deployment of agents based on this sample.
