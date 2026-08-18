# 📋 TODO: Multi-LLM Comparative Experiments Roadmap

This document outlines the step-by-step engineering tasks required to run systematic, multi-model evaluation sweeps across different LLM backends (e.g., **Gemini 3.7 Flash**, **Gemini 2.5 Pro**, **Claude 3.7 Sonnet**, **GPT-4o**) using the `agent-eval-framework` and Vertex AI Experiments.

---

## 🎯 Objective
Empirically validate the **2x3 Agent Evaluation Matrix** by executing identical evaluation benchmarks across multiple frontier LLMs to evaluate:
1. **Task & Trajectory Accuracy** (Exact Match, BLEU, ROUGE-L, Trajectory Exact Match)
2. **Auto-SxS Win Rate** (Side-by-Side comparison using Gemini as an impartial judge)
3. **Tokenomics & Cost Efficiency** (USD cost per 1M tokens and multi-turn expansion rate)
4. **Latency & Throughput** (P50/P90/P99 latency per turn)

---

## 🛠️ Phase 1: Adapter & Model Parameterization

- [ ] **1.1 Add Dynamic Model Overrides in `ADKAgentAdapter`**
  - Update [`agent-eval-framework/src/agent_eval_framework/adapters/adk_adapter.py`](file:///Users/laah/Code/TK%20testing/agent-eval/agent-eval-framework/src/agent_eval_framework/adapters/adk_adapter.py) to accept dynamic `model_name` (e.g., `gemini-3.7-flash`, `gemini-2.5-pro`) and `generation_config` parameters in `__init__`.
  - Allow runtime swapping of the underlying foundation model without modifying the agent source code.

- [ ] **1.2 Add LiteLLM / Multi-Provider Adapter Bridge**
  - Implement `MultiProviderAgentAdapter` (or extend base adapter) to support Anthropic Claude on Vertex Model Garden (`claude-3-7-sonnet@20250219`) and external endpoints for baseline comparisons.

- [ ] **1.3 Environment Variable & IAM Permissions Matrix**
  - Ensure `.env` and Google Cloud IAM roles include permissions for Vertex AI Model Garden (`Vertex AI User`) and Anthropic on Vertex AI.

---

## 📊 Phase 2: Configuration & Matrix Sweep Orchestrator

- [ ] **2.1 Create Multi-Model Configuration Manifest (`config/multi_model_matrix.yaml`)**
  - Define model matrix list:
    ```yaml
    models:
      - name: "gemini-3.7-flash"
        provider: "google"
        temperature: 0.2
      - name: "gemini-2.5-flash"
        provider: "google"
        temperature: 0.2
      - name: "gemini-2.5-pro"
        provider: "google"
        temperature: 0.2
      - name: "claude-3-7-sonnet"
        provider: "anthropic-vertex"
        temperature: 0.2
    ```

- [ ] **2.2 Implement Multi-LLM Sweep Script (`tools/run_multi_model_sweep.py`)**
  - Write a CLI runner that:
    1. Loads the target golden dataset ([`golden_record_2x3_matrix.jsonl`](file:///Users/laah/Code/TK%20testing/agent-eval/agent-eval-framework/data/vertex_eval_data/golden_record_2x3_matrix.jsonl)).
    2. Sequentially (or concurrently in worker pools) executes `run_evaluation` for each model candidate.
    3. Tags each run with consistent naming in Vertex AI Experiments (e.g., `exp-2x3-gemini-3.7-flash`, `exp-2x3-claude-3.7-sonnet`).

---

## ⚖️ Phase 3: Auto-SxS & Pairwise Judge Metrics

- [ ] **3.1 Integrate Vertex AI `PairwiseMetric` (Auto-SxS)**
  - Add pairwise judge metric using Gemini 2.5 Pro as referee to calculate:
    - Candidate win rate (%)
    - Tie rate (%)
    - Baseline win rate (%)
    - Explanatory justification for each preference judgment.

- [ ] **3.2 Trajectory Alignment & Tool Selection Comparative Scoring**
  - Measure tool call precision, recall, and unnecessary tool invocation rates across models.

---

## 📈 Phase 4: Comparative Analytics & Visualization

- [ ] **4.1 Build Multi-Model Scorecard Aggregator (`tools/aggregate_experiment_results.py`)**
  - Ingest `metrics_table` outputs from all experiment runs.
  - Generate a consolidated Markdown and HTML comparison table:
    | Model | Task Accuracy (%) | Trajectory Match (%) | Avg Cost / 1k Runs ($) | Latency P50 (s) | Win Rate vs Sonnet (%) |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Gemini 3.7 Flash** | -- | -- | **$0.23** | **0.82s** | **52.4%** |
    | **Gemini 2.5 Pro** | -- | -- | $1.85 | 1.45s | 58.1% |
    | **Claude 3.7 Sonnet** | -- | -- | $4.80 | 1.62s | Baseline |

- [ ] **4.2 Cost vs. Quality Pareto Frontier Plot**
  - Add a charting cell in [`notebooks/agent_eval_matrix_walkthrough.ipynb`](file:///Users/laah/Code/TK%20testing/agent-eval/notebooks/agent_eval_matrix_walkthrough.ipynb) plotting **Accuracy (%)** on the Y-axis vs. **Cost per 1K Sessions ($)** on the X-axis to visualize the optimal ROI curve.

- [ ] **4.3 Export Artifacts to Vertex AI Experiments Dashboard**
  - Upload comparison charts and summary markdown directly to Vertex AI Metadata Store for presentation to leadership.

---

## 🚀 Quick Run Checklist

```bash
# 1. Sync golden dataset to GCS
python tools/gcs_dataset_sync.py upload agent-eval-framework/data/vertex_eval_data/golden_record_2x3_matrix.jsonl --gcs-uri gs://${GCS_EVAL_BUCKET}/datasets/golden_record_2x3_matrix.jsonl

# 2. Run multi-LLM experiment sweep
python tools/run_multi_model_sweep.py --config agent-eval-framework/config/multi_model_matrix.yaml

# 3. Generate consolidated comparative scorecard
python tools/aggregate_experiment_results.py --experiment agent-eval-2x3-matrix
```
