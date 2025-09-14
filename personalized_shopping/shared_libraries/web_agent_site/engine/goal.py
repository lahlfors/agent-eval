# ... license headers ...
"""Functions for specifying goals and reward calculations."""

from collections import defaultdict
import itertools
import random
from rich import print
import spacy
from thefuzz import fuzz
# Absolute import
from .normalize import normalize_color

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' for spaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

PRICE_RANGE = [10.0 * i for i in range(1, 100)]

# ... rest of the file as you provided ...
# (get_goals, get_human_goals, get_synthetic_goals, get_type_reward, etc.)
