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

"""Converts raw product data into a format suitable for search indexing.

This script reads a JSON file containing a list of products, processes each
product, and transforms it into a structured format required by the search
engine. The output is a JSONL file where each line represents a single
document to be indexed. The 'contents' of each document are a concatenation
of key product fields (Title, Description, etc.) to enable full-text search.
"""

import json
import sys
import os
import pathlib
from tqdm import tqdm

# --- Add project root to sys.path ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Absolute import
from personalized_shopping.shared_libraries.web_agent_site.engine.engine import load_products

# --- CORRECTED FILE PATH ---
INPUT_FILEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web_agent_site", "data", "items_shuffle_1000.json"))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "resources_1k")

print(f"Loading products from: {INPUT_FILEPATH}")
if not os.path.exists(INPUT_FILEPATH):
    raise FileNotFoundError(f"Input data file not found: {INPUT_FILEPATH}")
all_products, *_ = load_products(filepath=INPUT_FILEPATH)
print(f"Loaded {len(all_products)} products.")

docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get("options", {})
    for option_name, option_contents in options.items():
        if isinstance(option_contents, list):
            option_contents_text = ", ".join(option_contents)
            option_texts.append(f"{option_name}: {option_contents_text}")
    option_text = ", and ".join(option_texts)

    doc = dict()
    doc["id"] = p.get("asin")
    doc["contents"] = " ".join(
        [
            p.get("Title", ""),
            p.get("Description", ""),
            p.get("BulletPoints", [""])[0],
            option_text,
        ]
    ).lower()
    doc["product"] = p
    docs.append(doc)

print(f"Writing {len(docs)} documents to {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "documents.jsonl"), "w+") as f:
    for doc in docs:
        f.write(json.dumps(doc) + "\n")

print("Conversion complete.")
