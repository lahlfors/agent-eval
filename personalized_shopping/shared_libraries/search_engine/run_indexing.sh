#!/bin/bash
# ... license headers ...

INDEX_DIR="indexes"
RESOURCES_DIR="resources_1k"
INDEX_1K="${INDEX_DIR}/indexes_1k"

mkdir -p "${INDEX_DIR}"
# resources_1k is created by the python script

echo "--- Indexing 1k products ---"
poetry run python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${RESOURCES_DIR}" \
  --index "${INDEX_1K}" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

echo "--- Indexing complete for 1k ---"
