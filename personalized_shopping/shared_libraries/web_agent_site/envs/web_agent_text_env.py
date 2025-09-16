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

from collections import defaultdict
import json
import random
import string
import time
from bs4 import BeautifulSoup
from bs4.element import Comment
from flask import Flask
import gymnasium as gym
from gymnasium.envs.registration import register
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
from personalized_shopping.shared_libraries.web_agent_site.utils import (
    DEFAULT_FILE_PATH,
    FEAT_CONV,
    FEAT_IDS,
    random_idx,
)
# --- End Corrected Absolute Imports ---


app = Flask(__name__)


@app.route('/<session_id>', methods=['GET', 'POST'])
def index(session_id):
    """Dummy route to allow url_for('index') to work."""
    pass


@app.route('/search_results/<session_id>/<keywords>/<page>', methods=['GET', 'POST'])
def search_results(session_id, keywords, page):
    """Dummy route to allow url_for('search_results') to work."""
    pass


@app.route('/item_page/<session_id>/<asin>/<keywords>/<page>/<options>', methods=['GET', 'POST'])
def item_page(session_id, asin, keywords, page, options):
    """Dummy route to allow url_for('item_page') to work."""
    pass


@app.route('/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>', methods=['GET', 'POST'])
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    """Dummy route to allow url_for('item_sub_page') to work."""
    pass


@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
def done(session_id, asin, options):
    """Dummy route to allow url_for('done') to work."""
    pass


class WebAgentTextEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""

    def __init__(
        self,
        observation_mode="html",
        file_path=DEFAULT_FILE_PATH,
        server=None,
        **kwargs,
    ):
        """Constructor for text environment

        Arguments:

        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = "http://127.0.0.1:3000"
        self.server = (
            SimServer(
                self.base_url,
                self.file_path,
                self.kwargs.get("filter_goals"),
                self.kwargs.get("limit_goals", -1),
                self.kwargs.get("num_products"),
                self.kwargs.get("human_goals", 0),
                self.kwargs.get("show_attrs", False),
            )
            if server is None
            else server
        )
        self.browser = SimBrowser(self.server)

        self.session = self.kwargs.get("session")
        self.session_prefix = self.kwargs.get("session_prefix")
        if self.kwargs.get("get_image", 0):
            self.feats = torch.load(FEAT_CONV)
            self.ids = torch.load(FEAT_IDS)
            self.ids = {url: idx for idx, url in enumerate(self.ids)}
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get("num_prev_obs", 0)
        self.num_prev_actions = self.kwargs.get("num_prev_actions", 0)
        self.instruction_text = ""
        self.text_to_clickable = {}
        self.reset()

    def step(self, action):
        """Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:

        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = {}
        self.get_available_actions()

        # Determine action type (click, search) and argument
        action_name, action_arg = parse_action(action)
        if action_arg is not None:
            action_arg = action_arg.lower()

        status = dict(reward=0.0, done=False)
        if action_name == "search" and action_arg is not None and action_arg != "":
            status = self.browser.search(action_arg)
        elif (
            action_name == "click"
            and self.text_to_clickable and action_arg in self.text_to_clickable.keys()
            and action_arg != "search"
        ):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            print(f"Action '{action}' is not valid. Available clickables: {list(self.text_to_clickable.keys())}")

        # Update observation, state with the new action
        ob = self.observation
        text_list = [ob]
        if self.num_prev_actions > 0:
            self.prev_actions.append(action)
            if len(self.prev_actions) > self.num_prev_actions:
                self.prev_actions.pop(0)
        if self.num_prev_obs > 0:
             # prev_obs stores the history of observations *before* the current step
            if len(self.prev_obs) > self.num_prev_obs:
                self.prev_obs.pop(0)

        for i in range(1, max(self.num_prev_obs, self.num_prev_actions) + 1):
            if self.num_prev_actions >= i and len(self.prev_actions) >= i:
                text_list.append(self.prev_actions[-i])
            if self.num_prev_obs >= i and len(self.prev_obs) >= i:
                text_list.append(self.prev_obs[-i])

        self.prev_obs.append(ob) # Add current observation to history for next step
        state = " [SEP] ".join(text_list[::-1])
        return state, status["reward"], status["done"], info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id="search_input")
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_="btn")
        product_links = html_obj.find_all(class_="product-link")
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f"{b.get_text()}".lower(): b for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get("value")
            if opt_value:
                 self.text_to_clickable[f"{opt_value}".lower()] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id="product-image")
        if image_url is not None:
            image_url = image_url["src"]
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text_tag = html_obj.find(id="instruction-text")
        if instruction_text_tag and instruction_text_tag.h4:
            return instruction_text_tag.h4.text
        return ""

    def _parse_html(self, html=None):
        """Returns web request result wrapped in BeautifulSoup object"""
        if html is None:
            html = self.state["html"]
        html_obj = BeautifulSoup(html, "html.parser")
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state["html"]
        if self.observation_mode == "html":
            return html
        elif self.observation_mode == "text":
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == "text_rich":
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == "url":
            return self.state["url"]
        else:
            raise ValueError(f"Observation mode {self.observation_mode} not supported.")

    @property
    def state(self):
        """State that includes all information."""
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            return " [SEP] ".join(t.strip() for t in visible_texts if t.strip())
        else:
            observation = ""
            for t in visible_texts:
                if not t.strip():
                    continue
                if t.parent.name == "button":  # button
                    processed_t = f"[button] {t.strip()} [button_]"
                elif t.parent.name == "label":  # options
                    processed_t = f"  [button] {t.strip()} [button_]"
                elif t.parent.get("class") == ["product-link"]:  # product asins
                     processed_t = f"\n[button] {t.strip()} [button_]"
                else:  # regular, unclickable text
                    processed_t = str(t).strip()
                observation += processed_t + "\n"
            return observation

    def reset(self, session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = "".join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session

        init_url = f"{self.base_url}/{self.session}"
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = {}
        self.instruction_text = (
            self.get_instruction_text()
            if instruction_text is None
            else instruction_text
        )
        if self.server:
            self.server.assigned_instruction_text = self.instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        return obs, {} # Return empty info dict

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def tag_visible(element):
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""

    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
    ):
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = (
            load_products(
                filepath=file_path,
                num_products=num_products,
                human_goals=human_goals,
            )
        )
        self.search_engine = init_search_engine(num_products=num_products)
        self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        self.show_attrs = show_attrs
        random.seed(233)
        random.shuffle(self.goals)
        if filter_goals is not None:
            self.goals = [
                goal for (i, goal) in enumerate(self.goals) if filter_goals(i, goal)
            ]
        if limit_goals != -1 and limit_goals < len(self.goals):
             self.weights = [goal["weight"] for goal in self.goals]
             self.cum_weights = [0] + np.cumsum(self.weights).tolist()
             idxs = []
             while len(idxs) < limit_goals:
                 idx = random_idx(self.cum_weights)
                 if idx not in idxs:
                     idxs.append(idx)
             self.goals = [self.goals[i] for i in idxs]
        print(f"Loaded {len(self.goals)} goals.")
        self.weights = [goal["weight"] for goal in self.goals]
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None

    def receive(self, session_id, current_url, session_int=None, **kwargs):
        status = dict(reward=0.0, done=False)
        with app.test_request_context():
            if session_id not in self.user_sessions:
                idx = session_int if session_int is not None else random_idx(self.cum_weights)
                goal = self.goals[idx % len(self.goals)]
                self.user_sessions[session_id] = {"goal": goal, "done": False, "actions": defaultdict(int), "asins": set(), "options": dict()}
            session = self.user_sessions[session_id]
            instruction_text = session["goal"]["instruction_text"]
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
            # self.assigned_instruction_text = instruction_text # This seems to be the only place it's set

            action_name = kwargs.get("action_name")
            text_to_clickable = kwargs.get("text_to_clickable", {})

            if not action_name:
                if not kwargs: # reset
                     action_name = "start"
                elif "keywords" in kwargs:
                    action_name = "search"
                elif "clickable_name" in kwargs:
                    action_name = "click"

            html = ""
            url = current_url
            if action_name == "start":
                html = map_action_to_html("start", session_id=session_id, instruction_text=instruction_text)
                url = f"{self.base_url}/{session_id}"
                session.update({"keywords": None, "page": None, "asin": None, "asins": set(), "options": dict(), "actions": defaultdict(int)})
            elif action_name == "search":
                keywords = kwargs["keywords"]
                page = kwargs.get("page", 1)
                session["page"] = page
                session["keywords"] = keywords
                session["actions"]["search"] += 1
                session["asin"] = None
                session["options"] = {}
                top_n_products = get_top_n_product_from_keywords(keywords, self.search_engine, self.all_products, self.product_item_dict)
                products = get_product_per_page(top_n_products, page)
                url = f"{self.base_url}/search_results/{session_id}/{'+'.join(keywords)}/{page}"
                html = map_action_to_html("search", session_id=session_id, products=products, keywords=keywords, page=page, total=len(top_n_products), instruction_text=instruction_text)
            elif action_name == "click":
                clickable_name = kwargs["clickable_name"]
                if clickable_name.lower() == END_BUTTON.lower():
                    product_info = self.product_item_dict[session["asin"]]
                    reward, info = get_reward(product_info, session["goal"], self.product_prices.get(session["asin"]), session["options"], verbose=True)
                    status["reward"], status["done"] = reward, True
                    session["done"], session["reward"], session["verbose_info"] = True, reward, info
                    url = f"{self.base_url}/done/{session_id}/{session['asin']}/{session['options']}"
                    html = map_action_to_html(f"click[{END_BUTTON}]", session_id=session_id, reward=reward, asin=session["asin"], options=session["options"], instruction_text=instruction_text)
                elif clickable_name.lower() == BACK_TO_SEARCH.lower():
                     return self.receive(session_id, current_url)
                elif clickable_name.lower() == NEXT_PAGE.lower() and self.get_page_name(current_url) == "search_results":
                    return self.receive(session_id, current_url, keywords=session["keywords"], page=session["page"] + 1, action_name="search")
                elif clickable_name.lower() == PREV_PAGE.lower() and self.get_page_name(current_url) == "search_results":
                     return self.receive(session_id, current_url, keywords=session["keywords"], page=session["page"] - 1, action_name="search")
                elif clickable_name.lower() == PREV_PAGE.lower() and self.get_page_name(current_url) == "item_sub_page":
                     html, url = self.item_page(session_id, session["asin"], session["keywords"], session["page"], session["options"], instruction_text, **kwargs)
                elif clickable_name.lower() == PREV_PAGE.lower() and self.get_page_name(current_url) == "item_page":
                     return self.receive(session_id, current_url, keywords=session["keywords"], page=session["page"], action_name="search")
                elif clickable_name.lower() in [k.lower() for k in ACTION_TO_TEMPLATE]:
                     product_info = self.product_item_dict[session["asin"]]
                     session["actions"][clickable_name] += 1
                     url = f"{self.base_url}/item_sub_page/{session_id}/{session['asin']}/{'+'.join(session['keywords'])}/{session['page']}/{clickable_name}/{session['options']}"
                     html = map_action_to_html(f"click[{clickable_name}]", session_id=session_id, product_info=product_info, keywords=session["keywords"], page=session["page"], asin=session["asin"], options=session["options"], instruction_text=instruction_text)
                else: # item page or option click
                     html, url = self.item_page(session_id, session["asin"], session["keywords"], session["page"], session["options"], instruction_text, **kwargs)
            else:
                raise ValueError(f"Invalid kwargs or action_name: {kwargs}")
            return html, url, status

    def item_page(self, session_id, asin, keywords, page, options, instruction_text, **kwargs):
         product_info = self.product_item_dict[asin]
         url = f"{self.base_url}/item_page/{session_id}/{asin}/{'+'.join(keywords)}/{page}/{options}"
         html = map_action_to_html("click", session_id=session_id, product_info=product_info, keywords=keywords, page=page, asin=asin, options=options, instruction_text=instruction_text, show_attrs=self.show_attrs)
         return html, url

    def get_page_name(self, url):
        if url is None: return None
        for page_name in ["search_results", "item_page", "item_sub_page", "done"]:
            if page_name in url: return page_name
        return ""

class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages."""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        self.session_id = url.split("/")[-1] if session_id is None else session_id
        self.page_source, self.current_url, _ = self.server.receive(self.session_id, self.current_url, session_int=session_int)

    def click(self, clickable_name, text_to_clickable):
        self.page_source, self.current_url, status = self.server.receive(
            self.session_id,
            current_url=self.current_url,
            clickable_name=clickable_name,
            text_to_clickable=text_to_clickable,
            action_name="click"
        )
        return status

    def search(self, keywords):
        if isinstance(keywords, str): keywords = keywords.split(" ")
        self.page_source, self.current_url, status = self.server.receive(
            self.session_id,
            current_url=self.current_url,
            keywords=keywords,
            action_name="search"
        )
        return status

# register(
#     id="WebAgentTextEnv-v0",
#     entry_point=(
#         "personalized_shopping.shared_libraries.web_agent_site.envs.web_agent_text_env:WebAgentTextEnv"
#     ),
# )
