"""
MCTS agent for Blackjack using determinization for partial observability
(dealer hole card unknown).
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent
from engine.deck import Card, ALL_CARDS, hand_value, is_bust, is_blackjack
from engine.game import ObservableState
from engine.rules import (
    ACTION_DOUBLE,
    ACTION_HIT,
    ACTION_SPLIT,
    ACTION_STAND,
    compute_payout,
    dealer_should_hit,
    get_legal_actions,
)


# ---------------------------------------------------------------------------
# Lightweight game state for MCTS rollouts
# ---------------------------------------------------------------------------

@dataclass
class MCTSGameState:
    """Minimal game state for tree search rollouts."""
    player_hand: list[Card]
    dealer_hand: list[Card]          # full (hole card known in simulation)
    deck_remaining: list[Card]
    bet: float
    game_over: bool = False
    payout: float = 0.0
    doubled: bool = False

    def copy(self) -> "MCTSGameState":
        return MCTSGameState(
            player_hand=self.player_hand[:],
            dealer_hand=self.dealer_hand[:],
            deck_remaining=self.deck_remaining[:],
            bet=self.bet,
            game_over=self.game_over,
            payout=self.payout,
            doubled=self.doubled,
        )


def _deal_from(deck: list[Card], rng: random.Random) -> Card:
    if not deck:
        return rng.choice(ALL_CARDS)
    idx = rng.randrange(len(deck))
    deck[idx], deck[-1] = deck[-1], deck[idx]
    return deck.pop()


def _finalize(state: MCTSGameState, soft17_hit: bool = True) -> float:
    """Run dealer to completion and return net payout (normalized by bet)."""
    while dealer_should_hit(state.dealer_hand, soft17_hit):
        if state.deck_remaining:
            state.dealer_hand.append(state.deck_remaining.pop())
        else:
            break
    p = compute_payout(state.player_hand, state.dealer_hand, state.bet)
    return p / max(state.bet, 1.0)


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    game_state: MCTSGameState
    action: str | None
    parent: "MCTSNode | None"
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.game_state.game_over

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1(self, parent_visits: int, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.total_reward / self.visits) + c * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def best_child(self, c: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(self.visits, c))


# ---------------------------------------------------------------------------
# MCTS Agent
# ---------------------------------------------------------------------------

class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent for Blackjack.

    Uses determinization to handle the hidden dealer hole card:
    1. Sample N plausible hole cards consistent with visible information.
    2. Run MCTS on each determinized world.
    3. Aggregate visit counts, pick action with most visits.
    """

    def __init__(
        self,
        player_id: int = 0,
        n_simulations: int = 200,
        n_determinizations: int = 10,
        ucb_c: float = 1.41,
        seed: int | None = None,
    ) -> None:
        super().__init__(player_id)
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
        self.ucb_c = ucb_c
        self.rng = random.Random(seed)

    def name(self) -> str:
        return f"MCTS(sims={self.n_simulations})"

    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        if len(legal_actions) == 1:
            self.last_reason = f"only one legal action: {legal_actions[0]}"
            return legal_actions[0]

        action_visits: dict[str, int] = {a: 0 for a in legal_actions}
        sims_per_world = max(1, self.n_simulations // self.n_determinizations)

        for _ in range(self.n_determinizations):
            world = self._sample_world(state)
            root = MCTSNode(
                game_state=world,
                action=None,
                parent=None,
                untried_actions=list(legal_actions),
            )
            self._run_mcts(root, sims_per_world)
            for child in root.children:
                if child.action in action_visits:
                    action_visits[child.action] += child.visits

        best = max(action_visits, key=lambda a: action_visits[a])
        total = sum(action_visits.values()) or 1
        best_pct = int(100 * action_visits[best] / total)
        self.last_reason = (
            f"MCTS: '{best}' chosen in {total} simulations ({best_pct}% visits)"
        )
        return best

    def _sample_world(self, state: ObservableState) -> MCTSGameState:
        """Build a determinized world by sampling a plausible hole card."""
        # Build pool of unknown cards
        known: set[Card] = set(state.player_hand) | {state.dealer_upcard}
        for c in state.seen_cards:
            known.add(c)

        unknown: list[Card] = [c for c in ALL_CARDS if c not in known]
        # Weight: multiple copies possible in shoe (not tracked precisely, sample uniformly)
        self.rng.shuffle(unknown)

        # Dealer hole card: pick from unknown pool
        hole_card = unknown[0] if unknown else self.rng.choice(ALL_CARDS)
        deck_remaining = unknown[1:] if len(unknown) > 1 else []

        return MCTSGameState(
            player_hand=list(state.player_hand),
            dealer_hand=[state.dealer_upcard, hole_card],
            deck_remaining=deck_remaining,
            bet=state.current_bet,
        )

    def _run_mcts(self, root: MCTSNode, n_iterations: int) -> None:
        for _ in range(n_iterations):
            node = self._select(root)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while (
            not node.is_terminal()
            and node.is_fully_expanded()
            and node.children
        ):
            node = node.best_child(self.ucb_c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            return node

        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)

        new_state = node.game_state.copy()
        new_state, next_legal = self._apply_action(new_state, action)

        child = MCTSNode(
            game_state=new_state,
            action=action,
            parent=node,
            untried_actions=next_legal,
        )
        node.children.append(child)
        return child

    def _apply_action(
        self, state: MCTSGameState, action: str
    ) -> tuple[MCTSGameState, list[str]]:
        """Apply action to state; return updated state and next legal actions."""
        if action == ACTION_STAND:
            # Dealer plays out
            while dealer_should_hit(state.dealer_hand):
                card = _deal_from(state.deck_remaining, self.rng)
                state.dealer_hand.append(card)
            state.payout = compute_payout(
                state.player_hand, state.dealer_hand, state.bet
            )
            state.game_over = True
            return state, []

        elif action == ACTION_HIT:
            card = _deal_from(state.deck_remaining, self.rng)
            state.player_hand.append(card)
            if is_bust(state.player_hand):
                state.payout = -state.bet
                state.game_over = True
                return state, []
            legal = get_legal_actions(
                state.player_hand,
                bankroll=1e9,
                current_bet=state.bet,
                can_double=False,
                can_split=False,
                is_first_action=False,
            )
            return state, legal

        elif action == ACTION_DOUBLE:
            state.doubled = True
            state.bet *= 2
            card = _deal_from(state.deck_remaining, self.rng)
            state.player_hand.append(card)
            if is_bust(state.player_hand):
                state.payout = -state.bet
                state.game_over = True
                return state, []
            # Exactly one card then stand
            while dealer_should_hit(state.dealer_hand):
                state.dealer_hand.append(_deal_from(state.deck_remaining, self.rng))
            state.payout = compute_payout(
                state.player_hand, state.dealer_hand, state.bet
            )
            state.game_over = True
            return state, []

        elif action == ACTION_SPLIT:
            # Simplified split: play first split hand only
            card1 = state.player_hand[0]
            state.player_hand = [card1, _deal_from(state.deck_remaining, self.rng)]
            legal = get_legal_actions(
                state.player_hand,
                bankroll=1e9,
                current_bet=state.bet,
                can_double=False,
                can_split=False,
                is_first_action=True,
            )
            return state, legal

        return state, []

    def _simulate(self, node: MCTSNode) -> float:
        """Random rollout to terminal; return normalized payout in [-1, 1]."""
        if node.is_terminal():
            return node.game_state.payout / max(node.game_state.bet, 1.0)

        state = node.game_state.copy()
        legal = list(node.untried_actions) + [
            c.action for c in node.children if c.action is not None
        ]
        if not legal:
            legal = [ACTION_STAND]

        while not state.game_over:
            # Exclude SPLIT/DOUBLE from rollout for simplicity
            simple = [a for a in legal if a in (ACTION_HIT, ACTION_STAND)]
            if not simple:
                simple = [ACTION_STAND]
            action = self.rng.choice(simple)
            state, legal = self._apply_action(state, action)
            if not legal and not state.game_over:
                # Force stand
                state, legal = self._apply_action(state, ACTION_STAND)

        return state.payout / max(state.bet, 1.0)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
