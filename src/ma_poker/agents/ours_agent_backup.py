# Backup of original OursAgent implementation
# This file contains the original single-agent version before refactoring to multi-agent architecture
# Date: Backup created before multi-agent refactoring

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ma_poker.agents.base import ActOutput, Agent
from ma_poker.agents.hand_utils import (
    cards_to_hand_type,
    get_top_hands,
    format_top_hands_for_prompt,
    INDEX_TO_HAND_TYPE,
)


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


def _format_cards(cards: List[str]) -> str:
    """Format card list to readable string."""
    if not cards:
        return "None"
    return ", ".join(cards)


# NOTE: Original implementation continues here...
# This is a backup file. See ours_agent.py for the current multi-agent implementation.
