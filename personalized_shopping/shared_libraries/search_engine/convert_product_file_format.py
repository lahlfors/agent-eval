# Copyright 2025 Google LLC
# ... (license headers) ...
"""Pre-processes product data for the search engine."""

import json
import sys
import os
import pathlib
from tqdm import tqdm

# --- Add project root to sys.path ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# print(f"Adjusted sys.path for convert_product_file_format: {sys.path}")

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
