"""Blackjack rules: actions, payouts, dealer logic."""
from __future__ import annotations

from engine.deck import Card, hand_value, is_bust, is_blackjack

# Player actions
ACTION_HIT = "hit"
ACTION_STAND = "stand"
ACTION_DOUBLE = "double"
ACTION_SPLIT = "split"

ALL_ACTIONS = (ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT)


def get_legal_actions(
    hand: list[Card],
    bankroll: float,
    current_bet: float,
    can_double: bool = True,
    can_split: bool = True,
    is_first_action: bool = True,
) -> list[str]:
    """
    Return list of legal actions for the current hand state.

    - HIT and STAND are always legal (unless already bust).
    - DOUBLE: legal on first action if player can afford it.
    - SPLIT: legal on first action if both cards share the same rank and player can afford it.
    """
    if is_bust(hand):
        return []

    actions = [ACTION_HIT, ACTION_STAND]

    if is_first_action and can_double and bankroll >= current_bet:
        actions.append(ACTION_DOUBLE)

    if (
        is_first_action
        and can_split
        and len(hand) == 2
        and hand[0].rank == hand[1].rank
        and bankroll >= current_bet
    ):
        actions.append(ACTION_SPLIT)

    return actions


def dealer_should_hit(hand: list[Card], soft17_hit: bool = True) -> bool:
    """
    Standard casino dealer rules:
    - Dealer hits on hard 16 or less.
    - Dealer hits on soft 17 if soft17_hit is True (common rule).
    - Dealer stands on hard 17+.
    """
    val = hand_value(hand)
    if val < 17:
        return True
    if val == 17 and soft17_hit:
        # Check if soft 17
        from engine.deck import is_soft
        return is_soft(hand)
    return False


def compute_payout(
    player_hand: list[Card],
    dealer_hand: list[Card],
    bet: float,
    blackjack_pays: float = 1.5,
) -> float:
    """
    Compute net payout (positive = player wins, negative = player loses).

    - Natural blackjack: pays 3:2 (blackjack_pays=1.5) unless dealer also has blackjack (push).
    - Normal win: pays 1:1.
    - Push: 0.
    - Loss: -bet.
    """
    player_bj = is_blackjack(player_hand)
    dealer_bj = is_blackjack(dealer_hand)

    player_val = hand_value(player_hand)
    dealer_val = hand_value(dealer_hand)

    player_bust = player_val > 21
    dealer_bust = dealer_val > 21

    if player_bust:
        return -bet

    if player_bj and dealer_bj:
        return 0.0  # push

    if player_bj:
        return bet * blackjack_pays

    if dealer_bj:
        return -bet

    if dealer_bust:
        return bet

    if player_val > dealer_val:
        return bet
    elif player_val == dealer_val:
        return 0.0  # push
    else:
        return -bet


def upcard_value(dealer_upcard: Card) -> int:
    """Dealer upcard value as seen by players (Ace = 11)."""
    return dealer_upcard.value()
