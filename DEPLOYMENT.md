# Deployment Guide: Local Agent with Cloud Evaluation

This guide provides the full, end-to-end process for setting up and running the `personalized_shopping` agent in a local environment on macOS. The agent itself will run entirely on your local machine, while the evaluation service will run on Google Cloud's Vertex AI.

This hybrid setup is ideal for development and testing, as it allows you to iterate on the agent locally while still leveraging the powerful, cloud-based evaluation metrics.

## Part 1: Local Environment Setup

These are the one-time setup steps required on your local Mac.

**Prerequisites:**
*   Python 3.12
*   Poetry for dependency management.
*   Homebrew for installing Java.

### 1.1: Install Python Dependencies

Navigate to the project root (`agent-eval/`) in your terminal and use Poetry to install the exact versions of all required libraries from the lock file.

```bash
# From the project root (agent-eval/)
poetry install
```

### 1.2: Install and Configure Java (Required for Search)

The agent's search functionality relies on Pyserini, which requires a Java Runtime Environment (JRE).

1.  **Install Java using Homebrew:**
    ```bash
    brew install openjdk
    ```

2.  **Link Java to the System Path:**
    Homebrew's installation is "keg-only." You must create a symbolic link so that macOS can find it.
    ```bash
    # This command requires administrator privileges
    sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
    ```

3.  **Set the `JAVA_HOME` Environment Variable:**
    Open your shell's configuration file (e.g., `~/.zshrc` or `~/.bash_profile`) and add the following line:
    ```bash
    # Add this line to your ~/.zshrc or ~/.bash_profile
    export JAVA_HOME=$(/usr/libexec/java_home)
    ```
    **Important:** Restart your terminal for this change to take effect.

## Part 2: Data Pipeline Setup

This section covers the download and processing of the product data required for the agent's simulated e-commerce environment.

### 2.1: Download Product Data

For local development, we will use the smaller "1000-item" sample files.

1.  **Install `gdown`:**
    This tool downloads files from Google Drive.
    ```bash
    pip install gdown
    ```

2.  **Download the data files:**
    ```bash
    # Navigate to the correct directory and create the 'data' folder
    cd personalized_shopping/shared_libraries
    mkdir data
    cd data

    # Download Main product catalog
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib

    # Download Human-written goals/attributes
    gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O

    # Download V2 goals/attributes
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
    ```

### 2.2: Build the Search Index

The raw JSON data must be processed into a format the search engine can use.

1.  **Navigate to the search engine directory:**
    From the `personalized_shopping/shared_libraries/data` directory, go back to the search engine directory.
    ```bash
    cd ../search_engine
    ```

2.  **Run the conversion and indexing scripts:**
    ```bash
    # Create necessary resource directories
    mkdir -p resources_1k

    # Convert the data format
    python convert_product_file_format.py

    # Build the search indexes
    bash run_indexing.sh
    ```
    *Note: These scripts are one-time utilities. They may remove themselves after successful execution, which is expected behavior.*

## Part 3: Environment Configuration for Cloud Evaluation

Even though the agent runs locally, the evaluation framework needs to communicate with Google Cloud's Vertex AI service.

1.  **Create a `.env` file:**
    In the project root, copy the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file:**
    Open the `.env` file and fill in your Google Cloud project details. The `AGENT_ENGINE_ID` is no longer needed for this local setup, but you must provide your project ID and region.
    ```
    # .env
    GCP_PROJECT_ID="your-gcp-project-id"
    GCP_REGION="your-gcp-region"
    AGENT_ENGINE_ID="" # Not used in local evaluation
    ```

Your local environment is now fully configured. See the `DEMO.md` guide for instructions on how to run the agent and the evaluation.
