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

"""Unit tests for GCS dataset synchronization and validation."""

import json
import tempfile
from pathlib import Path
import pytest
from tools.gcs_dataset_sync import (
    parse_gcs_uri,
    validate_dataset_file,
    upload_dataset,
    download_dataset,
)


def test_parse_gcs_uri():
    bucket, blob = parse_gcs_uri("gs://my-bucket/path/to/dataset.jsonl")
    assert bucket == "my-bucket"
    assert blob == "path/to/dataset.jsonl"

    bucket_only, blob_empty = parse_gcs_uri("gs://my-bucket")
    assert bucket_only == "my-bucket"
    assert blob_empty == ""

    with pytest.raises(ValueError):
        parse_gcs_uri("https://storage.googleapis.com/bucket/file")


def test_validate_dataset_file_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "Hello", "reference": "Hi there!", "reference_trajectory": []}) + "\n")
        f.write(json.dumps({"prompt": "Search", "reference_response": "Results", "reference_trajectory": [{"tool": "search"}]}) + "\n")
        temp_path = f.name

    is_valid, errors = validate_dataset_file(temp_path)
    Path(temp_path).unlink()
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataset_file_invalid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("invalid json string\n")
        f.write(json.dumps({"foo": "bar"}) + "\n")  # missing prompt and reference
        temp_path = f.name

    is_valid, errors = validate_dataset_file(temp_path)
    Path(temp_path).unlink()
    assert is_valid is False
    assert len(errors) > 0


def test_upload_dataset_mocked(mocker):
    mock_storage = mocker.patch("tools.gcs_dataset_sync.storage.Client")
    mock_bucket = mock_storage.return_value.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "Hi", "reference": "Hello"}) + "\n")
        temp_path = f.name

    success = upload_dataset(temp_path, "gs://test-bucket/evals/test.jsonl")
    Path(temp_path).unlink()

    assert success is True
    mock_blob.upload_from_filename.assert_called_once()
