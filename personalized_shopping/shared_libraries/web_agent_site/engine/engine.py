# Example in engine.py
import os
from google.cloud import storage
from pyserini.search.lucene import LuceneSearcher

class SearchEngine:
    def __init__(self, index_name="indexes_50k"):
        self.data_bucket_name = os.getenv("DATA_BUCKET")
        self.index_name = index_name
        self.local_index_path = f"/tmp/{self.index_name}"
        self._searcher = None  # Initialize searcher as None

        if not self.data_bucket_name:
            print("Warning: DATA_BUCKET not set, search will not function.")

    def _ensure_index_is_ready(self):
        if self._searcher:
            return

        if os.path.exists(self.local_index_path) and os.listdir(self.local_index_path):
            print(f"Index '{self.index_name}' found locally at {self.local_index_path}")
        else:
            if not self.data_bucket_name:
                raise ValueError("DATA_BUCKET env var not set, cannot download index.")
            print(f"Downloading index '{self.index_name}' from GCS bucket '{self.data_bucket_name}'...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.data_bucket_name)
            # GCS path example: gs://your-data-bucket/indexes/indexes_50k/
            gcs_prefix = f"indexes/{self.index_name}/"
            blobs = bucket.list_blobs(prefix=gcs_prefix)

            found_files = False
            for blob in blobs:
                if not blob.name.endswith('/'):
                    found_files = True
                    relative_path = os.path.relpath(blob.name, gcs_prefix)
                    destination_file_name = os.path.join(self.local_index_path, relative_path)
                    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                    # print(f"Downloading {blob.name} to {destination_file_name}")
                    blob.download_to_filename(destination_file_name)

            if not found_files:
                raise FileNotFoundError(f"No index files found in gs://{self.data_bucket_name}/{gcs_prefix}")
            print("Index download complete.")

        try:
            self._searcher = LuceneSearcher(self.local_index_path)
            print(f"Search engine initialized with index: {self.local_index_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Lucene index from {self.local_index_path}: {e}")

    def search(self, query: str):
        self._ensure_index_is_ready()
        if not self._searcher:
             raise RuntimeError("Searcher not initialized.")
        print(f"Searching for: {query}")
        hits = self._searcher.search(query)
        return hits

def init_search_engine(num_products=None):
    # NOTE: num_products seems unused in the original on error, adjust as needed
    # Return the SearchEngine instance, don't trigger download here.
    return SearchEngine()
