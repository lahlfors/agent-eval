# Copyright 2025 Google LLC
# ... (license headers) ...

from collections import defaultdict
import json
import random
import string
import time
from bs4 import BeautifulSoup
from bs4.element import Comment
from flask import Flask
import gym
from gym.envs.registration import register
import numpy as np
import torch

# --- Corrected Absolute Imports ---
from personalized_shopping.shared_libraries.web_agent_site.engine.engine import (
    ACTION_TO_TEMPLATE,
    BACK_TO_SEARCH,
    END_BUTTON,
    NEXT_PAGE,
    PREV_PAGE,
    get_product_per_page,
    get_top_n_product_from_keywords,
    init_search_engine,
    load_products,
    map_action_to_html,
    parse_action,
)
from personalized_shopping.shared_libraries.web_agent_site.engine.goal import get_goals, get_reward
from personalized_shopping.shared_libraries.web_agent_site.engine.utils import (
    DEFAULT_FILE_PATH,
    FEAT_CONV,
    FEAT_IDS,
    random_idx,
)
# --- End Corrected Absolute Imports ---

app = Flask(__name__)

# ... (WebAgentTextEnv, SimServer, SimBrowser classes as provided) ...
(The rest of the web_agent_text_env.py file is the same as the version you provided at 2025-09-14T15:26:54)
