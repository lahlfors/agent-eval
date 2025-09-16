# src/agent_eval_framework/utils/gcs_utils.py
import os
import tempfile
from urllib.parse import urlparse
import uuid
from google.cloud import storage
from .logging_utils import log

def download_gcs_file(gcs_path: str) -> str:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}.")
    parsed_url = urlparse(gcs_path)
    bucket_name = parsed_url.netloc
    blob_name = parsed_url.path.lstrip('/')
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS path: {gcs_path}.")
    try:
        project_id = os.getenv("GCP_PROJECT_ID")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            log.error(f"GCS object not found: {gcs_path}")
            raise FileNotFoundError(f"GCS object not found: {gcs_path}")
        temp_dir = tempfile.gettempdir()
        file_name = os.path.basename(blob_name) or f"downloaded{uuid.uuid4()}.tmp"
        temp_file_path = os.path.join(temp_dir, file_name)
        blob.download_to_filename(temp_file_path)
        log.info(f"Successfully downloaded {gcs_path} to {temp_file_path}")
        return temp_file_path
    except Exception as e:
        log.error(f"Failed to download {gcs_path}: {e}", exc_info=True)
        raise
