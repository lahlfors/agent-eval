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

"""Utility functions for the web shopping simulation environment.

This module provides helper functions for various tasks within the webshop
simulation, including weighted random sampling, logger setup, and MTurk code
generation. It also defines constants for file paths used throughout the
environment.
"""

import bisect
import hashlib
import logging
from os.path import abspath, dirname, join
import random
from pathlib import Path
from typing import List

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

DEFAULT_ATTR_PATH = join(BASE_DIR, "../data/items_ins_v2_1000.json")
DEFAULT_FILE_PATH = join(BASE_DIR, "../data/items_shuffle_1000.json")

DEFAULT_REVIEW_PATH = join(BASE_DIR, "../data/reviews.json")

FEAT_CONV = join(BASE_DIR, "../data/feat_conv.pt")
FEAT_IDS = join(BASE_DIR, "../data/feat_ids.pt")

HUMAN_ATTR_PATH = join(BASE_DIR, "../data/items_human_ins.json")


def random_idx(cum_weights: List[float]) -> int:
    """Selects a random index based on cumulative weights.

    This function performs weighted random sampling. It samples a uniform value
    from the total weight and uses binary search to find the corresponding index.

    Args:
        cum_weights: A list of cumulative weights.

    Returns:
        The selected random index.
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx


def setup_logger(session_id: str, user_log_dir: Path) -> logging.Logger:
    """Creates and configures a logger for a specific session.

    This function sets up a logger that writes to a session-specific JSONL file
    in the provided directory.

    Args:
        session_id: The unique identifier for the current session.
        user_log_dir: The directory where the log file should be created.

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(user_log_dir / f"{session_id}.jsonl", mode="w")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def generate_mturk_code(session_id: str) -> str:
    """Generates a unique MTurk completion code from a session ID.

    This creates a deterministic, short, and unique code that can be given to
    an Amazon Mechanical Turk worker to verify task completion.

    Args:
        session_id: The unique identifier for the session.

    Returns:
        A 10-character uppercase hexadecimal string.
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()
