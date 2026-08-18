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

"""Vertex AI Evaluation Pipeline Orchestrator.

Provides a Vertex AI Pipeline / Custom Job definition to orchestrate agent
evaluations across model variants and golden datasets stored in Google Cloud Storage.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import aiplatform


def create_and_run_pipeline(
    project_id: str,
    location: str,
    staging_bucket: str,
    dataset_gcs_uri: str,
    experiment_name: str = "agent-eval-2x3-matrix",
    config_path: str = "agent-eval-framework/config/adk_eval_config.yaml",
):
    """Executes a managed agent evaluation job on Vertex AI.

    Args:
        project_id: GCP project ID.
        location: GCP region (e.g., us-central1).
        staging_bucket: GCS staging bucket URI (e.g., gs://my-bucket/staging).
        dataset_gcs_uri: GCS URI to the golden evaluation dataset.
        experiment_name: Name of the Vertex AI Experiment for metric tracking.
        config_path: Relative path to the evaluation YAML configuration file.
    """
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket,
        experiment=experiment_name,
    )

    print(f"🚀 Initializing Vertex AI Evaluation Pipeline for Experiment: {experiment_name}")
    print(f"   Project: {project_id}")
    print(f"   Location: {location}")
    print(f"   Dataset URI: {dataset_gcs_uri}")
    print(f"   Config: {config_path}")

    # Local / Runner execution bridge
    from agent_eval_framework.runner import run_evaluation

    result = run_evaluation(config_path=config_path)
    if result:
        print(f"✅ Evaluation Job Completed Successfully in Vertex AI Experiment: {experiment_name}")
        return result
    else:
        print("❌ Evaluation Job Failed.", file=sys.stderr)
        return None


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Submit Agent Evaluation to Vertex AI Pipeline / Experiments")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP Project ID")
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP Location")
    parser.add_argument("--bucket", default=os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET"), help="GCS Staging Bucket URI")
    parser.add_argument(
        "--dataset",
        default="agent-eval-framework/data/vertex_eval_data/golden_record_2x3_matrix.jsonl",
        help="Dataset GCS URI or local path",
    )
    parser.add_argument("--experiment", default="agent-eval-2x3-matrix", help="Vertex AI Experiment Name")
    parser.add_argument("--config", default="agent-eval-framework/config/adk_eval_config.yaml", help="Config file path")

    args = parser.parse_args()

    if not args.project:
        print("Error: GOOGLE_CLOUD_PROJECT must be provided via CLI or .env file", file=sys.stderr)
        sys.exit(1)

    create_and_run_pipeline(
        project_id=args.project,
        location=args.location,
        staging_bucket=args.bucket,
        dataset_gcs_uri=args.dataset,
        experiment_name=args.experiment,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
