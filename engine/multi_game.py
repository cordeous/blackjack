"""
Multi-agent Blackjack game engine.

All agents play against the SAME dealer hand each round.
The shoe is shared — cards dealt to players and dealer come from the same deck.
Each agent has its own bankroll and bet independently.

Usage:
    game = MultiAgentGame(agents=[agent1, agent2, ...], ...)
    result = game.run()
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engine.deck import Card, Deck, hand_value, is_bust, is_blackjack
from engine.rules import (
    ACTION_DOUBLE,
    ACTION_HIT,
    ACTION_SPLIT,
    ACTION_STAND,
    compute_payout,
    dealer_should_hit,
    get_legal_actions,
)
from engine.game import ObservableState, RoundRecord

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Per-agent round data
# ---------------------------------------------------------------------------

@dataclass
class AgentRoundStep:
    """One decision taken by one agent."""
    agent_name: str
    agent_index: int
    player_hand: list[Card]
    dealer_upcard: Card
    legal_actions: list[str]
    action_taken: str
    reason: str
    hand_value: int
    is_split_hand: bool = False
    hand_index: int = 0


@dataclass
class AgentRoundResult:
    """Full outcome for one agent in one round."""
    agent_name: str
    agent_index: int
    player_hands: list[list[Card]]
    bets: list[float]
    payouts: list[float]           # net payout per hand
    actions_taken: list[list[str]]
    bankroll_after: float
    steps: list[AgentRoundStep] = field(default_factory=list)

    @property
    def net_payout(self) -> float:
        return sum(self.payouts)

    @property
    def won(self) -> bool:
        return self.net_payout > 0

    @property
    def lost(self) -> bool:
        return self.net_payout < 0

    @property
    def pushed(self) -> bool:
        return self.net_payout == 0


@dataclass
class MultiRoundRecord:
    """Complete record of one round across all agents."""
    round_num: int
    dealer_hand: list[Card]
    dealer_upcard: Card
    agent_results: list[AgentRoundResult]   # one per agent (same order as agents list)


@dataclass
class AgentLeaderboardEntry:
    """Cumulative stats for one agent across all rounds."""
    name: str
    agent_index: int
    starting_bankroll: float
    final_bankroll: float
    wins: int = 0
    losses: int = 0
    ties: int = 0
    blackjacks: int = 0
    total_payout: float = 0.0
    # Point system: win=3pts, tie=1pt, loss=0pts, blackjack bonus=2pts
    points: int = 0

    @property
    def rounds_played(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.rounds_played, 1)

    @property
    def net_profit(self) -> float:
        return self.final_bankroll - self.starting_bankroll

    @property
    def rank(self) -> int:
        return 0  # filled in after sorting


@dataclass
class MultiSessionResult:
    """Full result of a multi-agent session."""
    rounds_played: int
    starting_bankroll: float
    rounds: list[MultiRoundRecord]
    leaderboard: list[AgentLeaderboardEntry]  # sorted by points desc


# ---------------------------------------------------------------------------
# Multi-agent game
# ---------------------------------------------------------------------------

class MultiAgentGame:
    """
    Runs N agents against the same dealer each round.

    All agents share the same shoe (cards are dealt sequentially:
    player1 cards, player2 cards, ..., dealer cards). This mirrors
    real casino play where one shoe serves all seats.
    """

    def __init__(
        self,
        agents: list["BaseAgent"],
        num_rounds: int = 20,
        starting_bankroll: float = 1000.0,
        base_bet: float = 10.0,
        num_decks: int = 6,
        seed: int | None = None,
        soft17_hit: bool = True,
        blackjack_pays: float = 1.5,
    ) -> None:
        self.agents = agents
        self.num_rounds = num_rounds
        self.starting_bankroll = starting_bankroll
        self.base_bet = base_bet
        self.soft17_hit = soft17_hit
        self.blackjack_pays = blackjack_pays
        self.deck = Deck(num_decks=num_decks, seed=seed)
        self.bankrolls: list[float] = [starting_bankroll] * len(agents)
        self.seen_cards: list[Card] = []
        self.round_records: list[MultiRoundRecord] = []
        self.round_num = 0

    def run(self) -> MultiSessionResult:
        for agent in self.agents:
            agent.reset()

        for _ in range(self.num_rounds):
            # Stop agents who can't afford to play
            any_can_play = any(
                self.bankrolls[i] >= self.base_bet
                for i in range(len(self.agents))
            )
            if not any_can_play:
                break

            if self.deck.is_low():
                self.deck.shuffle()
                self.seen_cards.clear()

            self._play_round()

        leaderboard = self._build_leaderboard()
        return MultiSessionResult(
            rounds_played=len(self.round_records),
            starting_bankroll=self.starting_bankroll,
            rounds=self.round_records,
            leaderboard=leaderboard,
        )

    def _deal_card(self) -> Card:
        card = self.deck.deal()
        self.seen_cards.append(card)
        return card

    def _play_round(self) -> None:
        self.round_num += 1
        n = len(self.agents)

        # Deal initial hands: each agent gets 2 cards, dealer gets 2 cards
        player_hands: list[list[Card]] = []
        for _ in range(n):
            player_hands.append([self._deal_card(), self._deal_card()])

        dealer_hand: list[Card] = [self._deal_card(), self._deal_card()]
        # Hide hole card from seen
        self.seen_cards.pop()
        hole_card = dealer_hand[1]
        dealer_upcard = dealer_hand[0]

        dealer_natural = is_blackjack(dealer_hand)
        agent_results: list[AgentRoundResult] = []

        for i, agent in enumerate(self.agents):
            if self.bankrolls[i] < self.base_bet:
                # Agent is broke — record a skip
                result = AgentRoundResult(
                    agent_name=agent.name(),
                    agent_index=i,
                    player_hands=[player_hands[i][:]],
                    bets=[0.0],
                    payouts=[0.0],
                    actions_taken=[["broke"]],
                    bankroll_after=self.bankrolls[i],
                    steps=[],
                )
                agent_results.append(result)
                continue

            bet = self.base_bet
            self.bankrolls[i] -= bet
            steps: list[AgentRoundStep] = []

            player_hand = player_hands[i]

            # Natural blackjack check
            if is_blackjack(player_hand):
                if dealer_natural:
                    payout = 0.0
                else:
                    payout = bet * self.blackjack_pays
                self.bankrolls[i] += bet + payout
                result = AgentRoundResult(
                    agent_name=agent.name(),
                    agent_index=i,
                    player_hands=[player_hand[:]],
                    bets=[bet],
                    payouts=[payout],
                    actions_taken=[["blackjack"]],
                    bankroll_after=self.bankrolls[i],
                    steps=steps,
                )
                agent_results.append(result)
                continue

            if dealer_natural:
                # Player loses
                self.bankrolls[i] += 0  # bet already removed
                result = AgentRoundResult(
                    agent_name=agent.name(),
                    agent_index=i,
                    player_hands=[player_hand[:]],
                    bets=[bet],
                    payouts=[-bet],
                    actions_taken=[["dealer_blackjack"]],
                    bankroll_after=self.bankrolls[i],
                    steps=steps,
                )
                agent_results.append(result)
                continue

            # Normal play
            hands: list[list[Card]] = [player_hand]
            bets: list[float] = [bet]
            actions_taken: list[list[str]] = [[]]
            hand_idx = 0

            while hand_idx < len(hands):
                current_hand = hands[hand_idx]
                current_bet = bets[hand_idx]
                is_split = hand_idx > 0 or len(hands) > 1
                hand_actions = actions_taken[hand_idx]
                is_first_action = True

                while not is_bust(current_hand):
                    legal = get_legal_actions(
                        current_hand,
                        bankroll=self.bankrolls[i],
                        current_bet=current_bet,
                        can_double=is_first_action and not is_split,
                        can_split=is_first_action and not is_split,
                        is_first_action=is_first_action,
                    )
                    if not legal:
                        break

                    obs = ObservableState(
                        player_hand=current_hand[:],
                        dealer_upcard=dealer_upcard,
                        deck_cards_remaining=self.deck.cards_remaining(),
                        bankroll=self.bankrolls[i],
                        current_bet=current_bet,
                        is_first_action=is_first_action,
                        seen_cards=self.seen_cards[:],
                        round_num=self.round_num,
                        is_split_hand=is_split,
                        split_hand_index=hand_idx,
                        num_split_hands=len(hands),
                    )

                    action = agent.choose_action(obs, legal)
                    reason = getattr(agent, "last_reason", "")
                    hand_actions.append(action)

                    step = AgentRoundStep(
                        agent_name=agent.name(),
                        agent_index=i,
                        player_hand=current_hand[:],
                        dealer_upcard=dealer_upcard,
                        legal_actions=list(legal),
                        action_taken=action,
                        reason=reason,
                        hand_value=hand_value(current_hand),
                        is_split_hand=is_split,
                        hand_index=hand_idx,
                    )
                    steps.append(step)

                    if action == ACTION_STAND:
                        break
                    elif action == ACTION_HIT:
                        current_hand.append(self._deal_card())
                        is_first_action = False
                    elif action == ACTION_DOUBLE:
                        self.bankrolls[i] -= current_bet
                        bets[hand_idx] = current_bet * 2
                        current_bet = bets[hand_idx]
                        current_hand.append(self._deal_card())
                        break
                    elif action == ACTION_SPLIT:
                        card1, card2 = current_hand[0], current_hand[1]
                        self.bankrolls[i] -= current_bet
                        hands[hand_idx] = [card1, self._deal_card()]
                        new_hand = [card2, self._deal_card()]
                        hands.insert(hand_idx + 1, new_hand)
                        bets.insert(hand_idx + 1, current_bet)
                        actions_taken.insert(hand_idx + 1, [])
                        current_hand = hands[hand_idx]
                        current_bet = bets[hand_idx]
                        is_split = True
                        is_first_action = True
                        continue

                hand_idx += 1

            # Compute payouts (dealer plays after all agents done)
            payouts: list[float] = []
            for h, b in zip(hands, bets):
                # Note: dealer plays once after all agents act (below)
                # For now store hands; payouts computed after dealer plays
                payouts.append(0.0)  # placeholder

            result = AgentRoundResult(
                agent_name=agent.name(),
                agent_index=i,
                player_hands=hands,
                bets=bets,
                payouts=payouts,
                actions_taken=actions_taken,
                bankroll_after=0.0,  # filled after dealer plays
                steps=steps,
            )
            agent_results.append(result)

        # Dealer plays (reveal hole card, draw to 17+)
        self.seen_cards.append(hole_card)
        while dealer_should_hit(dealer_hand, self.soft17_hit):
            dealer_hand.append(self._deal_card())

        # Finalize payouts now that dealer is done
        for i, result in enumerate(agent_results):
            if result.actions_taken == [["broke"]]:
                continue
            if result.actions_taken[0][0] in ("blackjack", "dealer_blackjack"):
                # Already finalized above
                continue

            total_payout = 0.0
            for hi, (h, b) in enumerate(zip(result.player_hands, result.bets)):
                p = compute_payout(h, dealer_hand, b, self.blackjack_pays)
                result.payouts[hi] = p
                total_payout += p
                self.bankrolls[i] += b + p

            result.bankroll_after = self.bankrolls[i]

        # Fix bankroll_after for blackjack/dealer_blackjack cases (already correct)
        for i, result in enumerate(agent_results):
            if result.actions_taken[0][0] in ("blackjack", "dealer_blackjack"):
                result.bankroll_after = self.bankrolls[i]

        rec = MultiRoundRecord(
            round_num=self.round_num,
            dealer_hand=dealer_hand[:],
            dealer_upcard=dealer_upcard,
            agent_results=agent_results,
        )
        self.round_records.append(rec)

    def _build_leaderboard(self) -> list[AgentLeaderboardEntry]:
        entries: list[AgentLeaderboardEntry] = []
        for i, agent in enumerate(self.agents):
            entry = AgentLeaderboardEntry(
                name=agent.name(),
                agent_index=i,
                starting_bankroll=self.starting_bankroll,
                final_bankroll=self.bankrolls[i],
            )
            for rec in self.round_records:
                if i >= len(rec.agent_results):
                    continue
                ar = rec.agent_results[i]
                net = ar.net_payout
                entry.total_payout += net

                # Detect blackjack
                bj = any("blackjack" in acts for acts in ar.actions_taken
                         if acts and acts[0] == "blackjack")
                if bj:
                    entry.blackjacks += 1

                if net > 0:
                    entry.wins += 1
                    entry.points += 3
                    if bj:
                        entry.points += 2  # blackjack bonus
                elif net == 0:
                    entry.ties += 1
                    entry.points += 1
                else:
                    entry.losses += 1

            entries.append(entry)

        # Sort by points descending, then by win rate as tiebreaker
        entries.sort(key=lambda e: (e.points, e.win_rate, e.net_profit), reverse=True)
        return entries
