# Copyright 2025 Google LLC
# ... (license headers) ...
"""Functions for specifying goals and reward calculations."""

from collections import defaultdict
import itertools
import random
from rich import print
import spacy
from thefuzz import fuzz
# Absolute import
from personalized_shopping.shared_libraries.web_agent_site.engine.normalize import normalize_color

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' for spaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

PRICE_RANGE = [10.0 * i for i in range(1, 100)]


def get_goals(all_products, product_prices, human_goals=True):
    if human_goals:
        return get_human_goals(all_products, product_prices)
    else:
        return get_synthetic_goals(all_products, product_prices)


def get_human_goals(all_products, product_prices):
    goals = []
    cnt_atts = defaultdict(int)
    cnt = 0
    for item in all_products:
        asin = item["asin"]
        if "instructions" not in item:
            continue
        for product in item["instructions"]:
            attributes = product["instruction_attributes"]
            if len(attributes) == 0:
                cnt += 1
                continue

            price_upper = 1000000
            price_text = ""
            if product_prices is not None:
                price = product_prices.get(asin)
                if price is not None:
                    price_range = [p for p in PRICE_RANGE if p > price][:4]
                    if len(price_range) >= 2:
                        _, price_upper = sorted(random.sample(price_range, 2))
                        price_text = f", and price lower than {price_upper:.2f} dollars"

            goals.append(
                {
                    "asin": asin,
                    "category": item["category"],
                    "query": item["query"],
                    "name": item["name"],
                    "product_category": item["product_category"],
                    "instruction_text": product["instruction"].strip(".") + price_text,
                    "attributes": attributes,
                    "price_upper": price_upper,
                    "goal_options": product["instruction_options"],
                }
            )
            for att in attributes:
                cnt_atts[att] += 1
    for goal in goals:
        goal["weight"] = 1
    print(f"{cnt} items skipped due to no attributes in human goals")
    return goals


def get_synthetic_goals(all_products, product_prices):
    # ... (This function looks OK, no imports to change) ...
    goals = []
    cnt_atts = defaultdict(int)
    for product in all_products:
        if "instruction_text" not in product or product["instruction_text"] is None:
            continue
        product_goals = []
        asin = product["asin"]
        attributes = product["instruction_attributes"]
        assert len(attributes) > 0

        price_upper = 1000000
        price_text = ""
        if product_prices is not None:
            price = product_prices.get(asin)
            if price is not None:
                price_range = [p for p in PRICE_RANGE if p > price][:4]
                if len(price_range) >= 2:
                    _, price_upper = sorted(random.sample(price_range, 2))
                    price_text = f", and price lower than {price_upper:.2f} dollars"

        instruction_text = product["instruction_text"]

        options = product["options"]
        option_names = sorted(options)
        option_values_list = [options[name] for name in option_names]

        if not all(option_values_list): # Skip if any option list is empty
            continue

        combinations = list(itertools.product(*option_values_list))

        for combination in combinations:
            goal_options = dict(zip(option_names, combination))
            option_text = ", and ".join([f"{k}: {v}" for k, v in goal_options.items()])
            option_text = " with " + option_text if option_text else ""
            product_goals.append(
                {
                    "asin": asin,
                    "category": product["category"],
                    "query": product["query"],
                    "name": product["name"],
                    "product_category": product["product_category"],
                    "instruction_text": f"{instruction_text}{option_text}{price_text}",
                    "attributes": attributes,
                    "price_upper": price_upper,
                    "goal_options": goal_options,
                    "title": product["Title"],
                }
            )
            for att in attributes:
                cnt_atts[att] += 1
        goals += product_goals
    if not goals: return [] # Avoid division by zero if no goals
    for goal in goals:
        if not goal["attributes"]: # Avoid division by zero
             goal["weight"] = 0
             continue
        goal["weight"] = sum(1.0 / cnt_atts[att] for att in goal["attributes"] if cnt_atts[att] > 0) / len(
            goal["attributes"]
        )
    return goals


def get_type_reward(purchased_product, goal):
    """Determines the type reward - captures whether chosen product is in the same category"""
    query_match = purchased_product["query"] == goal["query"]

    purchased_product_category = [
        x.strip() for x in purchased_product["product_category"].split("›")
    ]
    goal_product_category = [x.strip() for x in goal["product_category"].split("›")]
    category_match = (
        len(set(purchased_product_category) & set(goal_product_category)) >= 2
    )

    purchased_type = purchased_product["name"]
    desired_type = goal["name"]

    purchased_type_parse = nlp(purchased_type)
    desired_type_parse = nlp(desired_type)

    purchased_type_parse = [
        t.text.lower()
        for t in purchased_type_parse
        if t.pos_ in ("PNOUN", "NOUN", "PROPN")
    ]
    desired_type_parse = [
        t.text.lower()
        for t in desired_type_parse
        if t.pos_ in ("PNOUN", "NOUN", "PROPN")
    ]

    n_intersect_type = len(set(purchased_type_parse) & set(desired_type_parse))
    if len(desired_type_parse) == 0:
        title_score = 0.2
    else:
        title_score = n_intersect_type / len(desired_type_parse)

    r_type = 1.0
    match = query_match or category_match or title_score > 0.2
    if not match:
        r_type = 0.5
    if title_score < 0.1:
        r_type = 0.1
    if title_score == 0.0:
        r_type = 0.0

    return dict(
        r_type=r_type,
        query_match=query_match,
        category_match=category_match,
        title_score=title_score,
    )


def get_attribute_reward(purchased_product, goal):
    """Determines whether purchased products shares same attributes as goal"""
    purchased_attrs = purchased_product["Attributes"]
    goal_attrs = goal["attributes"]
    if not goal_attrs: return 0.0, 0

    num_attr_matches = 0
    for g_attr in goal_attrs:
        matched = False
        for p_attr in purchased_attrs:
            score = fuzz.token_set_ratio(p_attr, g_attr)
            if score > 85:
                num_attr_matches += 1
                matched = True
                break
        if not matched and (
            g_attr in purchased_product["Title"].lower()
            or g_attr in " ".join(purchased_product["BulletPoints"]).lower()
            or g_attr in purchased_product["Description"].lower()
        ):
            num_attr_matches += 1
        #     matched = True # This was missing

    r_attr = num_attr_matches / len(goal_attrs)
    return r_attr, num_attr_matches


def get_option_reward(purchased_options, goal_options):
    """Calculate reward for purchased product's options w.r.t. goal options"""
    if not goal_options: return None, 0
    if isinstance(goal_options, dict):
         goal_options = list(goal_options.values())

    purchased_options = [normalize_color(str(o)) for o in purchased_options]
    goal_options = [normalize_color(str(o)) for o in goal_options]

    num_option_matches = 0
    for g_option in goal_options:
        for p_option in purchased_options:
            score = fuzz.token_set_ratio(p_option, g_option)
            if score > 85:
                num_option_matches += 1
                break
    r_option = num_option_matches / len(goal_options)
    return r_option, num_option_matches


def get_reward(purchased_product, goal, price, options, **kwargs):
    """Get cumulative reward score for purchased product and goal"""
    r_type_dict = get_type_reward(purchased_product, goal)

    r_price = (price <= goal["price_upper"]) if goal.get("price_upper") is not None and price is not None else 0.0

    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)

    goal_options = goal.get("goal_options", {})
    r_option, num_option_matches = get_option_reward(
        list(options.values()),
        goal_options
    )

    option_len = len(goal_options)
    attr_len = len(goal["attributes"])
    denominator = attr_len + option_len + 1
    if denominator == 0: return 0.0

    if r_option is None: r_option = 0
    if r_price is None: r_price = 0

    total_reward = (num_attr_matches + num_option_matches + float(r_price)) / denominator
    total_reward *= r_type_dict["r_type"]

    if kwargs.get("verbose", False):
        info = {
            "r_type": r_type_dict["r_type"],
            "r_att": r_att,
            "w_att": attr_len / denominator,
            "query_match": r_type_dict["query_match"],
            "category_match": r_type_dict["category_match"],
            "title_score": r_type_dict["title_score"],
            "r_option": r_option,
            "w_option": option_len / denominator,
            "r_price": r_price,
            "w_price": 1 / denominator,
        }
        return total_reward, info
    return total_reward
