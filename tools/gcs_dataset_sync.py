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

"""Google Cloud Storage (GCS) Golden Dataset Synchronizer.

This module provides programmatic and CLI tools to validate, upload, download,
and inspect evaluation datasets stored in GCS buckets (e.g. gs://bucket/path/dataset.jsonl).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage


def validate_dataset_file(file_path: str) -> Tuple[bool, List[str]]:
    """Validates the schema and formatting of a JSONL evaluation dataset file.

    Args:
        file_path: Path to the local JSON or JSONL dataset file.

    Returns:
        A tuple of (is_valid, list_of_error_messages).
    """
    path = Path(file_path)
    if not path.exists():
        return False, [f"File not found: {file_path}"]

    errors = []
    records = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        errors.append(f"Line {idx}: Record is not a valid JSON object/dict.")
                        continue
                    records.append(record)
                except json.JSONDecodeError as err:
                    errors.append(f"Line {idx}: JSON parse error: {err}")
    except Exception as e:
        return False, [f"Failed to read file {file_path}: {e}"]

    if not records and not errors:
        return False, ["Dataset file is empty."]

    # Validate individual record schema
    for idx, record in enumerate(records, 1):
        # Check prompt field (or query)
        has_prompt = "prompt" in record or "query" in record
        if not has_prompt:
            errors.append(f"Record {idx}: Missing required 'prompt' (or 'query') field.")

        # Check reference field (or reference_response / target)
        has_ref = (
            "reference" in record
            or "reference_response" in record
            or "reference_answer" in record
            or "target" in record
        )
        if not has_ref:
            errors.append(f"Record {idx}: Missing expected reference response ('reference' or 'reference_response').")

        # Optional check for reference_trajectory
        if "reference_trajectory" in record:
            traj = record["reference_trajectory"]
            if not isinstance(traj, (list, str)):
                errors.append(f"Record {idx}: 'reference_trajectory' must be a list or JSON string.")

    is_valid = len(errors) == 0
    return is_valid, errors


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """Splits a GCS URI (gs://bucket/blob/path) into bucket and blob name.

    Args:
        gcs_uri: The full GCS URI string.

    Returns:
        Tuple of (bucket_name, blob_name).

    Raises:
        ValueError: If uri format is invalid.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Expected format gs://bucket_name/path")
    stripped = gcs_uri[5:]
    if "/" not in stripped:
        return stripped, ""
    bucket_name, blob_path = stripped.split("/", 1)
    return bucket_name, blob_path


def upload_dataset(
    local_path: str,
    gcs_uri: str,
    validate: bool = True,
    project: Optional[str] = None,
) -> bool:
    """Uploads a local evaluation golden dataset to a GCS bucket.

    Args:
        local_path: Local path to the evaluation dataset (.jsonl).
        gcs_uri: Target GCS URI (e.g., gs://my-eval-bucket/datasets/golden_v1.jsonl).
        validate: Whether to validate the dataset schema before uploading.
        project: Google Cloud project ID (optional).

    Returns:
        True if upload succeeded, False otherwise.
    """
    if validate:
        is_valid, errors = validate_dataset_file(local_path)
        if not is_valid:
            print(f"❌ Validation failed for {local_path}:")
            for err in errors:
                print(f"   - {err}")
            return False
        print(f"✅ Dataset {local_path} passed schema validation.")

    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    if not blob_path:
        blob_path = Path(local_path).name

    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        print(f"🚀 Successfully uploaded {local_path} -> gs://{bucket_name}/{blob_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}", file=sys.stderr)
        return False


def download_dataset(
    gcs_uri: str,
    local_path: str,
    project: Optional[str] = None,
) -> str:
    """Downloads an evaluation dataset from GCS to a local file.

    Args:
        gcs_uri: Source GCS URI (e.g., gs://my-eval-bucket/datasets/golden_v1.jsonl).
        local_path: Local destination file path.
        project: Google Cloud project ID (optional).

    Returns:
        The path to the downloaded local file.
    """
    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    if not blob_path:
        raise ValueError(f"No file path specified in GCS URI: {gcs_uri}")

    dest_path = Path(local_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(dest_path))
        print(f"✅ Downloaded {gcs_uri} -> {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download from GCS: {e}", file=sys.stderr)
        raise


def list_gcs_datasets(
    gcs_prefix: str,
    project: Optional[str] = None,
) -> List[str]:
    """Lists dataset files under a GCS bucket prefix.

    Args:
        gcs_prefix: GCS URI prefix (e.g., gs://my-eval-bucket/datasets/).
        project: Google Cloud project ID (optional).

    Returns:
        List of matching GCS URIs.
    """
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        uris = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
        return uris
    except Exception as e:
        print(f"❌ Failed to list GCS bucket contents: {e}", file=sys.stderr)
        return []


def main():
    """CLI entrypoint for GCS dataset management."""
    parser = argparse.ArgumentParser(description="GCS Dataset Synchronization Tool for Agent Eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate subcommand
    val_parser = subparsers.add_parser("validate", help="Validate local dataset schema")
    val_parser.add_argument("file", help="Path to local .jsonl dataset file")

    # Upload subcommand
    up_parser = subparsers.add_parser("upload", help="Upload local dataset to GCS")
    up_parser.add_argument("file", help="Path to local .jsonl dataset file")
    up_parser.add_argument("--gcs-uri", required=True, help="Destination GCS URI (gs://bucket/path.jsonl)")
    up_parser.add_argument("--no-validate", action="store_true", help="Skip schema validation")
    up_parser.add_argument("--project", default=None, help="GCP project ID")

    # Download subcommand
    down_parser = subparsers.add_parser("download", help="Download dataset from GCS")
    down_parser.add_argument("--gcs-uri", required=True, help="Source GCS URI (gs://bucket/path.jsonl)")
    down_parser.add_argument("--output", required=True, help="Local destination file path")
    down_parser.add_argument("--project", default=None, help="GCP project ID")

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List datasets in GCS bucket")
    list_parser.add_argument("prefix", help="GCS URI prefix (gs://bucket/path/)")
    list_parser.add_argument("--project", default=None, help="GCP project ID")

    args = parser.parse_args()

    if args.command == "validate":
        valid, errors = validate_dataset_file(args.file)
        if valid:
            print(f"✅ Dataset {args.file} is valid!")
            sys.exit(0)
        else:
            print(f"❌ Validation errors in {args.file}:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)

    elif args.command == "upload":
        success = upload_dataset(
            local_path=args.file,
            gcs_uri=args.gcs_uri,
            validate=not args.no_validate,
            project=args.project,
        )
        sys.exit(0 if success else 1)

    elif args.command == "download":
        download_dataset(gcs_uri=args.gcs_uri, local_path=args.output, project=args.project)
        sys.exit(0)

    elif args.command == "list":
        results = list_gcs_datasets(gcs_prefix=args.prefix, project=args.project)
        print(f"Found {len(results)} dataset(s):")
        for uri in results:
            print(f"  - {uri}")
        sys.exit(0)


if __name__ == "__main__":
    main()
