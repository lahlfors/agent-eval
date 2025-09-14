# Copyright 2025 Google LLC
# ... (license headers) ...

"""Engine for WebShop Environment"""

from ast import literal_eval
from collections import defaultdict
from decimal import Decimal
import json
import os
import random
import re
from os.path import join, dirname, abspath

from flask import render_template_string
from pyserini.search.lucene import LuceneSearcher
from rich import print
from tqdm import tqdm

# Absolute imports
from personalized_shopping.shared_libraries.web_agent_site.engine.utils import (
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH,
)

BASE_DIR = dirname(abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = "Buy Now"
NEXT_PAGE = "Next >"
PREV_PAGE = "< Prev"
BACK_TO_SEARCH = "Back to Search"

ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    template_name = ""
    if action_name == "start":
        template_name = "search_page.html"
    elif action_name == "search":
        template_name = "results_page.html"
    elif action_name == "click":
        if action_arg == END_BUTTON:
            template_name = "done_page.html"
        elif action_arg in ACTION_TO_TEMPLATE:
            template_name = ACTION_TO_TEMPLATE[action_arg]
        else:
            template_name = "item_page.html"
    else:
        raise ValueError(f"Action name not recognized: {action}")

    path = os.path.join(TEMPLATE_DIR, template_name)
    html = render_template_string(read_html_template(path), **kwargs)
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """Parse action string to action name and its arguments."""
    pattern = re.compile(r"(.+)\[(.+)\]")
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == "keywords":
        keywords = string
        if keywords.startswith("["):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == "page":
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError("Name of variable not recognized.")
    return var


def get_top_n_product_from_keywords(
    keywords,
    search_engine,
    all_products,
    product_item_dict,
    attribute_to_asins=None,
):
    if keywords[0] == "<r>":
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == "<a>":
        attribute = " ".join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p["asin"] in asins]
    elif keywords[0] == "<c>":
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p["category"] == category]
    elif keywords[0] == "<q>":
        query = " ".join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p["query"] == query]
    else:
        keywords = " ".join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())["id"] for doc in docs]
        top_n_products = [
            product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict
        ]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW : page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product["asin"]
        pricing = product["pricing"]
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    if num_products == 100:
        index_suffix = "indexes_100"
    elif num_products == 1000:
        index_suffix = "indexes_1k"
    elif num_products == 10000:
        index_suffix = "indexes_10k"
    elif num_products == 50000:
        index_suffix = "indexes_50k"
    else:  # Default to 1k
        index_suffix = "indexes_1k"

    engine_dir = dirname(abspath(__file__))
    search_engine_root = abspath(os.path.join(engine_dir, "..", "..", "search_engine"))
    index_path = os.path.join(search_engine_root, "indexes", index_suffix)

    print(f"Initializing LuceneSearcher with index path: {index_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Search index not found at {index_path}. "
            f"Please run the indexing scripts in {search_engine_root}"
        )
    search_engine = LuceneSearcher(index_path)
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop("product_information", None)
        product.pop("brand", None)
        product.pop("brand_url", None)
        product.pop("list_price", None)
        product.pop("availability_quantity", None)
        product.pop("availability_status", None)
        product.pop("total_reviews", None)
        product.pop("total_answered_questions", None)
        product.pop("seller_id", None)
        product.pop("seller_name", None)
        product.pop("fulfilled_by_amazon", None)
        product.pop("fast_track_message", None)
        product.pop("aplus_present", None)
        product.pop("small_description_old", None)
    print("Keys cleaned.")
    return products


def load_products(filepath, num_products=None, human_goals=True):
    print(f"Attempting to load products from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Product file not found: {filepath}. Please check download and paths.")

    with open(filepath) as f:
        products = json.load(f)
    print("Products loaded.")
    products = clean_product_keys(products)

    all_reviews = dict()
    all_ratings = dict()

    human_attributes = {}
    if human_goals:
        if not os.path.exists(HUMAN_ATTR_PATH):
             print(f"Warning: Human attributes file not found: {HUMAN_ATTR_PATH}")
        else:
            with open(HUMAN_ATTR_PATH) as f:
                human_attributes = json.load(f)
    if not os.path.exists(DEFAULT_ATTR_PATH):
        raise FileNotFoundError(f"Default attributes file not found: {DEFAULT_ATTR_PATH}")
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    print("Attributes loaded.")

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        products = products[:num_products]

    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p.get("asin")
        if not asin or asin == "nan" or len(asin) > 10:
            continue
        if asin in asins:
            continue
        asins.add(asin)

        p["category"] = p.get("category", "")
        p["query"] = p.get("query", "")
        p["product_category"] = p.get("product_category", "")

        p["Title"] = p.get("name", "")
        p["Description"] = p.get("full_description", "")
        p["Reviews"] = all_reviews.get(asin, [])
        p["Rating"] = all_ratings.get(asin, "N.A.")
        for r in p["Reviews"]:
            if "score" not in r:
                r["score"] = r.pop("stars", None)
            if "review" not in r:
                r["body"] = ""
            else:
                r["body"] = r.pop("review")
        p["BulletPoints"] = (
            p.get("small_description", [])
            if isinstance(p.get("small_description"), list)
            else [p.get("small_description", "")]
        )

        pricing = p.get("pricing")
        price_tag = ""
        if not pricing:
            pricing = [100.0]
            price_tag = "$100.00"
        else:
            try:
                parsed_pricing = [
                    float(Decimal(re.sub(r"[^\d.]", "", price)))
                    for price in str(pricing).split("$") if re.sub(r"[^\d.]", "", price)
                ]
                if not parsed_pricing: parsed_pricing = [100.0]
                pricing = parsed_pricing
            except Exception:
                pricing = [100.0]

            if len(pricing) == 1:
                price_tag = f"${pricing[0]:.2f}"
            elif len(pricing) > 1:
                price_tag = f"${pricing[0]:.2f} to ${pricing[1]:.2f}"
                pricing = pricing[:2]
            else:
                 price_tag = "$100.00"
                 pricing = [100.0]

        p["pricing"] = pricing
        p["Price"] = price_tag

        options = dict()
        customization_options = p.get("customization_options")
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None: continue
                option_name = option_name.lower()
                option_values = []
                if isinstance(option_contents, list):
                    for option_content in option_contents:
                        if not isinstance(option_content, dict): continue
                        option_value = option_content.get("value", "").strip().replace("/", " | ").lower()
                        if not option_value: continue
                        option_image = option_content.get("image", None)
                        option_values.append(option_value)
                        option_to_image[option_value] = option_image
                options[option_name] = option_values
        p["options"] = options
        p["option_to_image"] = option_to_image

        if asin in attributes and "attributes" in attributes[asin]:
            p["Attributes"] = attributes[asin]["attributes"]
        else:
            p["Attributes"] = ["DUMMY_ATTR"]

        if human_goals:
            if asin in human_attributes:
                p["instructions"] = human_attributes[asin]
        else:
            p["instruction_text"] = attributes.get(asin, {}).get("instruction", None)
            p["instruction_attributes"] = attributes.get(asin, {}).get(
                "instruction_attributes", None
            )

        p["MainImage"] = p.get("images", [""])[0]
        p["query"] = p.get("query", "").lower().strip()

        all_products.append(p)

    for p in all_products:
        for a in p["Attributes"]:
            attribute_to_asins[a].add(p["asin"])

    product_item_dict = {p["asin"]: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
