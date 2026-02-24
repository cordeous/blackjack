"""
Heuristic agent implementing standard Blackjack basic strategy.

Basic strategy is the mathematically optimal play for each (player_total, dealer_upcard)
combination, based on millions of simulated hands.

Two modes:
  - "basic":      Follow standard basic strategy tables exactly
  - "aggressive": Always try to reach 17+; double/split more liberally
"""
from __future__ import annotations

from engine.deck import hand_value, is_soft
from engine.game import ObservableState
from engine.rules import (
    ACTION_DOUBLE,
    ACTION_HIT,
    ACTION_SPLIT,
    ACTION_STAND,
)
from agents.base_agent import BaseAgent


def _dealer_up(state: ObservableState) -> int:
    """Dealer upcard value, treating Ace as 11."""
    return state.dealer_upcard.value()


def _basic_strategy(
    player_hand: list,
    dealer_up: int,
    legal_actions: list[str],
    can_split: bool,
    can_double: bool,
) -> tuple[str, str]:
    """
    Return (action, reason) following standard basic strategy.
    Source: standard casino basic strategy (6-deck, dealer hits soft 17).
    """
    val = hand_value(player_hand)
    soft = is_soft(player_hand)
    pair = (
        len(player_hand) == 2
        and player_hand[0].rank == player_hand[1].rank
    )
    rank = player_hand[0].rank if pair else None

    # --- PAIR SPLITTING ---
    if pair and can_split and ACTION_SPLIT in legal_actions:
        # Always split Aces and 8s
        if rank == "A":
            return ACTION_SPLIT, "split Aces: always split"
        if rank == "8":
            return ACTION_SPLIT, "split 8s: always split"
        # Split 9s unless dealer shows 7, 10, or Ace
        if rank == "9" and dealer_up not in (7, 10, 11):
            return ACTION_SPLIT, f"split 9s vs dealer {dealer_up}"
        # Split 7s vs 2–7
        if rank == "7" and 2 <= dealer_up <= 7:
            return ACTION_SPLIT, f"split 7s vs dealer {dealer_up}"
        # Split 6s vs 2–6
        if rank == "6" and 2 <= dealer_up <= 6:
            return ACTION_SPLIT, f"split 6s vs dealer {dealer_up}"
        # Split 4s only vs 5–6 (if doubling allowed after split: approximate)
        if rank == "4" and dealer_up in (5, 6):
            return ACTION_SPLIT, f"split 4s vs dealer {dealer_up}"
        # Split 2s and 3s vs 2–7
        if rank in ("2", "3") and 2 <= dealer_up <= 7:
            return ACTION_SPLIT, f"split {rank}s vs dealer {dealer_up}"
        # Never split 5s (treat as hard 10), never split 10s

    # --- SOFT HANDS (Ace counted as 11) ---
    if soft:
        # Soft 19 (A+8) or better: stand
        if val >= 19:
            return ACTION_STAND, f"soft {val}: stand"
        # Soft 18
        if val == 18:
            if dealer_up in (2, 7, 8):
                return ACTION_STAND, f"soft 18 vs {dealer_up}: stand"
            if dealer_up in (3, 4, 5, 6) and can_double and ACTION_DOUBLE in legal_actions:
                return ACTION_DOUBLE, f"soft 18 vs {dealer_up}: double"
            return ACTION_HIT, f"soft 18 vs {dealer_up}: hit"
        # Soft 17 (A+6)
        if val == 17:
            if dealer_up in (3, 4, 5, 6) and can_double and ACTION_DOUBLE in legal_actions:
                return ACTION_DOUBLE, f"soft 17 vs {dealer_up}: double"
            return ACTION_HIT, f"soft 17: always hit"
        # Soft 13–16
        if val in (15, 16):
            if dealer_up in (4, 5, 6) and can_double and ACTION_DOUBLE in legal_actions:
                return ACTION_DOUBLE, f"soft {val} vs {dealer_up}: double"
            return ACTION_HIT, f"soft {val}: hit"
        if val in (13, 14):
            if dealer_up in (5, 6) and can_double and ACTION_DOUBLE in legal_actions:
                return ACTION_DOUBLE, f"soft {val} vs {dealer_up}: double"
            return ACTION_HIT, f"soft {val}: hit"
        return ACTION_HIT, f"soft {val}: hit"

    # --- HARD HANDS ---
    if val >= 17:
        return ACTION_STAND, f"hard {val}: always stand"
    if val == 16:
        if dealer_up >= 7:
            return ACTION_HIT, f"hard 16 vs {dealer_up}: hit (dealer strong)"
        return ACTION_STAND, f"hard 16 vs {dealer_up}: stand (dealer weak)"
    if val == 15:
        if dealer_up >= 7:
            return ACTION_HIT, f"hard 15 vs {dealer_up}: hit"
        return ACTION_STAND, f"hard 15 vs {dealer_up}: stand"
    if val in (13, 14):
        if dealer_up >= 7:
            return ACTION_HIT, f"hard {val} vs {dealer_up}: hit"
        return ACTION_STAND, f"hard {val} vs {dealer_up}: stand"
    if val == 12:
        if dealer_up in (4, 5, 6):
            return ACTION_STAND, f"hard 12 vs {dealer_up}: stand (dealer busts often)"
        return ACTION_HIT, f"hard 12 vs {dealer_up}: hit"
    if val == 11:
        if can_double and ACTION_DOUBLE in legal_actions:
            return ACTION_DOUBLE, "hard 11: always double"
        return ACTION_HIT, "hard 11: hit (no double available)"
    if val == 10:
        if dealer_up <= 9 and can_double and ACTION_DOUBLE in legal_actions:
            return ACTION_DOUBLE, f"hard 10 vs {dealer_up}: double"
        return ACTION_HIT, f"hard 10 vs {dealer_up}: hit"
    if val == 9:
        if dealer_up in (3, 4, 5, 6) and can_double and ACTION_DOUBLE in legal_actions:
            return ACTION_DOUBLE, f"hard 9 vs {dealer_up}: double"
        return ACTION_HIT, f"hard 9: hit"
    # 8 or less: always hit
    return ACTION_HIT, f"hard {val}: hit (low total)"


class HeuristicAgent(BaseAgent):
    """
    Basic strategy agent with two modes.

    basic:      Follows standard casino basic strategy exactly.
    aggressive: Hits until 17+, doubles on 10/11, always splits Aces/8s.
    """

    def __init__(self, player_id: int = 0, mode: str = "basic") -> None:
        if mode not in ("basic", "aggressive"):
            raise ValueError(f"mode must be 'basic' or 'aggressive', got '{mode}'")
        super().__init__(player_id)
        self.mode = mode

    def name(self) -> str:
        return f"Heuristic({self.mode})"

    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        if self.mode == "basic":
            return self._basic(state, legal_actions)
        return self._aggressive(state, legal_actions)

    def _basic(self, state: ObservableState, legal_actions: list[str]) -> str:
        can_split = ACTION_SPLIT in legal_actions
        can_double = ACTION_DOUBLE in legal_actions
        action, reason = _basic_strategy(
            state.player_hand,
            _dealer_up(state),
            legal_actions,
            can_split=can_split,
            can_double=can_double,
        )
        # Ensure action is legal
        if action not in legal_actions:
            action = ACTION_HIT if ACTION_HIT in legal_actions else ACTION_STAND
            reason += " (fallback)"
        self.last_reason = reason
        return action

    def _aggressive(self, state: ObservableState, legal_actions: list[str]) -> str:
        val = hand_value(state.player_hand)
        dealer_up = _dealer_up(state)

        # Always split Aces and 8s
        if (
            ACTION_SPLIT in legal_actions
            and len(state.player_hand) == 2
            and state.player_hand[0].rank == state.player_hand[1].rank
        ):
            rank = state.player_hand[0].rank
            if rank in ("A", "8"):
                self.last_reason = f"aggressive: always split {rank}s"
                return ACTION_SPLIT

        # Double on 10 or 11
        if ACTION_DOUBLE in legal_actions and val in (10, 11):
            self.last_reason = f"aggressive: double on {val}"
            return ACTION_DOUBLE

        # Hit until 17+
        if val < 17:
            self.last_reason = f"aggressive: hit on {val} (target 17+)"
            return ACTION_HIT

        self.last_reason = f"aggressive: stand on {val}"
        return ACTION_STAND
