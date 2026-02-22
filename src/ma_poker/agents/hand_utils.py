"""Utilities for Texas Hold'em hand representation and 169 hand types."""

from typing import Dict, List, Tuple


# 169种起手牌类型映射
# 格式: (hand_string, type) -> hand_type_index (0-168)
# 其中 hand_string 如 "AA", "AKs", "AKo" 等

HAND_TYPES = []
# 13种对子 (0-12): AA, KK, QQ, JJ, TT, 99, 88, 77, 66, 55, 44, 33, 22
for r in "AKQJT98765432":
    HAND_TYPES.append((f"{r}{r}", "pair"))

# 78种同花 (13-90): AKs, AQs, ..., 32s
rank_order = "AKQJT98765432"
for i, r1 in enumerate(rank_order):
    for r2 in rank_order[i+1:]:
        HAND_TYPES.append((f"{r1}{r2}s", "suited"))

# 78种不同花 (91-168): AKo, AQo, ..., 32o
for i, r1 in enumerate(rank_order):
    for r2 in rank_order[i+1:]:
        HAND_TYPES.append((f"{r1}{r2}o", "offsuit"))

# 反向映射: 手牌字符串 -> 索引
HAND_TYPE_TO_INDEX: Dict[str, int] = {hand: idx for idx, (hand, _) in enumerate(HAND_TYPES)}

# 索引 -> 手牌字符串
INDEX_TO_HAND_TYPE: Dict[int, str] = {idx: hand for idx, (hand, _) in enumerate(HAND_TYPES)}


def cards_to_hand_type(card1: str, card2: str) -> int:
    """Convert two cards to 169 hand type index.
    
    Args:
        card1: First card (e.g., "SA" for Spade Ace)
        card2: Second card (e.g., "HK" for Heart King)
    
    Returns:
        Hand type index (0-168)
    """
    # Extract rank and suit
    def parse_card(card: str) -> Tuple[str, str]:
        """Parse card string to (suit, rank)."""
        if len(card) < 2:
            return ("", "")
        suit = card[0]  # S, H, D, C
        rank = card[1:]  # A, K, Q, J, T, 9-2
        return (suit, rank)
    
    suit1, rank1 = parse_card(card1)
    suit2, rank2 = parse_card(card2)
    
    # Normalize ranks (A > K > Q > J > T > 9 > ... > 2)
    rank_order = "AKQJT98765432"
    
    def rank_value(r: str) -> int:
        return rank_order.index(r) if r in rank_order else 13
    
    val1, val2 = rank_value(rank1), rank_value(rank2)
    
    # Ensure val1 >= val2
    if val1 < val2:
        val1, val2 = val2, val1
        suit1, suit2 = suit2, suit1
    
    # Check if pair
    if rank1 == rank2:
        # Pair: use rank1 (higher rank)
        hand_str = f"{rank1}{rank2}"
        return HAND_TYPE_TO_INDEX.get(hand_str, 0)
    
    # Check if suited
    suited = (suit1 == suit2)
    suffix = "s" if suited else "o"
    hand_str = f"{rank1}{rank2}{suffix}"
    
    return HAND_TYPE_TO_INDEX.get(hand_str, 0)


def hand_type_to_string(hand_type_idx: int) -> str:
    """Convert hand type index to readable string.
    
    Args:
        hand_type_idx: Index (0-168)
    
    Returns:
        Hand string (e.g., "AA", "AKs", "AKo")
    """
    return INDEX_TO_HAND_TYPE.get(hand_type_idx, "??")


def get_top_hands(belief: List[float], top_k: int = 10) -> List[Tuple[int, float, str]]:
    """Get top K hands from belief distribution.
    
    Args:
        belief: Probability distribution over 169 hand types
        top_k: Number of top hands to return
    
    Returns:
        List of (index, probability, hand_string) tuples, sorted by probability descending
    """
    if len(belief) != 169:
        raise ValueError(f"Belief must have 169 elements, got {len(belief)}")
    
    # Get indices sorted by probability
    indexed_probs = [(i, prob) for i, prob in enumerate(belief)]
    indexed_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    top_hands = []
    for idx, prob in indexed_probs[:top_k]:
        hand_str = hand_type_to_string(idx)
        top_hands.append((idx, prob, hand_str))
    
    return top_hands


def format_top_hands_for_prompt(top_hands: List[Tuple[int, float, str]]) -> str:
    """Format top hands for inclusion in LLM prompt.
    
    Args:
        top_hands: List of (index, probability, hand_string) tuples
    
    Returns:
        Formatted string for prompt
    """
    if not top_hands:
        return "No hand information available."
    
    lines = ["Opponent's most likely hands (top 10):"]
    for i, (idx, prob, hand_str) in enumerate(top_hands, 1):
        lines.append(f"{i}. {hand_str}: {prob:.2%}")
    
    return "\n".join(lines)

