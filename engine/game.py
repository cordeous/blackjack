"""Blackjack game engine — single-player vs dealer, multi-round sessions."""
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

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """Complete record of one played round."""
    round_num: int
    player_hands: list[list[Card]]      # may have 2 hands after split
    dealer_hand: list[Card]
    bets: list[float]                    # one per player hand
    payouts: list[float]                 # net payout per hand
    actions_taken: list[list[str]]       # actions per hand
    bankroll_after: float


@dataclass
class ObservableState:
    """
    What the agent can observe at decision time.
    (Dealer hole card is hidden — only upcard visible.)
    """
    player_hand: list[Card]              # current hand being played
    dealer_upcard: Card                  # dealer's face-up card (hole card hidden)
    deck_cards_remaining: int
    bankroll: float
    current_bet: float
    is_first_action: bool               # True if no cards added to hand yet
    seen_cards: list[Card]              # all cards visible so far this shoe
    round_num: int
    # Split context
    is_split_hand: bool = False
    split_hand_index: int = 0
    num_split_hands: int = 1


@dataclass
class SessionResult:
    """Result of a full multi-round session."""
    rounds_played: int
    starting_bankroll: float
    final_bankroll: float
    net_profit: float
    rounds: list[RoundRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        wins = sum(1 for r in self.rounds if sum(r.payouts) > 0)
        return wins / max(len(self.rounds), 1)

    @property
    def avg_profit_per_round(self) -> float:
        return self.net_profit / max(self.rounds_played, 1)


# ---------------------------------------------------------------------------
# Game / Session
# ---------------------------------------------------------------------------

class BlackjackGame:
    """
    Orchestrates a Blackjack session (multiple rounds) with one agent vs dealer.

    Rules:
    - 6-deck shoe, reshuffle when shoe < 25% remaining
    - Dealer stands on hard 17, hits on soft 17
    - Natural blackjack pays 3:2
    - Double down on any first two cards
    - Split pairs (re-split not allowed, no double after split)
    - No surrender, no insurance
    """

    def __init__(
        self,
        agent: "BaseAgent",
        num_rounds: int = 100,
        starting_bankroll: float = 1000.0,
        base_bet: float = 10.0,
        num_decks: int = 6,
        seed: int | None = None,
        soft17_hit: bool = True,
        blackjack_pays: float = 1.5,
    ) -> None:
        self.agent = agent
        self.num_rounds = num_rounds
        self.starting_bankroll = starting_bankroll
        self.base_bet = base_bet
        self.soft17_hit = soft17_hit
        self.blackjack_pays = blackjack_pays
        self.deck = Deck(num_decks=num_decks, seed=seed)
        self.bankroll = starting_bankroll
        self.seen_cards: list[Card] = []
        self.round_records: list[RoundRecord] = []
        self.round_num = 0

    def run(self) -> SessionResult:
        self.agent.reset()
        for _ in range(self.num_rounds):
            if self.bankroll < self.base_bet:
                break
            if self.deck.is_low():
                self.deck.shuffle()
                self.seen_cards.clear()
            self._play_round()

        return SessionResult(
            rounds_played=len(self.round_records),
            starting_bankroll=self.starting_bankroll,
            final_bankroll=self.bankroll,
            net_profit=self.bankroll - self.starting_bankroll,
            rounds=self.round_records,
        )

    def _deal_card(self) -> Card:
        card = self.deck.deal()
        self.seen_cards.append(card)
        return card

    def _play_round(self) -> None:
        self.round_num += 1

        # Deal initial hands
        player_hand: list[Card] = [self._deal_card(), self._deal_card()]
        dealer_hand: list[Card] = [self._deal_card(), self._deal_card()]
        # Dealer hole card not added to seen until reveal
        self.seen_cards.pop()   # remove hole card from seen (hidden)
        hole_card = dealer_hand[1]
        dealer_upcard = dealer_hand[0]

        bet = self.base_bet
        self.bankroll -= bet

        # Check for dealer natural (check before player acts)
        dealer_natural = is_blackjack(dealer_hand)

        # Check for player natural
        if is_blackjack(player_hand):
            # Reveal hole card
            self.seen_cards.append(hole_card)
            if dealer_natural:
                payout = 0.0  # push
            else:
                payout = bet * self.blackjack_pays
            self.bankroll += bet + payout
            rec = RoundRecord(
                round_num=self.round_num,
                player_hands=[player_hand[:]],
                dealer_hand=dealer_hand[:],
                bets=[bet],
                payouts=[payout],
                actions_taken=[["blackjack"]],
                bankroll_after=self.bankroll,
            )
            self.round_records.append(rec)
            return

        # If dealer has natural, player loses immediately
        if dealer_natural:
            self.seen_cards.append(hole_card)
            payout = -bet
            self.bankroll += bet + payout  # == bankroll unchanged (-bet net)
            # Actually bankroll -= bet was done above, and payout = -bet means net is -bet
            # but we already subtracted bet; payout is net so just add payout back
            self.bankroll += payout  # double-subtract, fix: recalc below
            # Recalculate: player bet was removed, dealer wins, player gets 0 back
            # Reset: bankroll += 0  (bet lost entirely)
            # Fix the math: undo bankroll -= bet (already done) then apply payout
            # bankroll is currently: original - bet
            # payout = -bet, so net = -bet: correct, player gets 0 back
            # bankroll after = (original - bet), which is correct for a loss
            # But we added payout again, which is wrong — let me fix this properly
            # bankroll was: original - bet  (player put bet on table)
            # After loss, bankroll stays at: original - bet
            # Undo the extra payout add:
            self.bankroll -= payout  # undo the mistaken add
            rec = RoundRecord(
                round_num=self.round_num,
                player_hands=[player_hand[:]],
                dealer_hand=dealer_hand[:],
                bets=[bet],
                payouts=[-bet],
                actions_taken=[["dealer_blackjack"]],
                bankroll_after=self.bankroll,
            )
            self.round_records.append(rec)
            return

        # Player actions (possibly multiple hands after split)
        hands: list[list[Card]] = [player_hand]
        bets: list[float] = [bet]
        actions_taken: list[list[str]] = [[]]
        hand_idx = 0

        while hand_idx < len(hands):
            current_hand = hands[hand_idx]
            current_bet = bets[hand_idx]
            is_split_hand = hand_idx > 0 or len(hands) > 1
            hand_actions = actions_taken[hand_idx]

            is_first_action = True
            while not is_bust(current_hand):
                legal = get_legal_actions(
                    current_hand,
                    bankroll=self.bankroll,
                    current_bet=current_bet,
                    can_double=is_first_action and not is_split_hand,
                    can_split=is_first_action and not is_split_hand,
                    is_first_action=is_first_action,
                )
                if not legal:
                    break

                obs = ObservableState(
                    player_hand=current_hand[:],
                    dealer_upcard=dealer_upcard,
                    deck_cards_remaining=self.deck.cards_remaining(),
                    bankroll=self.bankroll,
                    current_bet=current_bet,
                    is_first_action=is_first_action,
                    seen_cards=self.seen_cards[:],
                    round_num=self.round_num,
                    is_split_hand=is_split_hand,
                    split_hand_index=hand_idx,
                    num_split_hands=len(hands),
                )

                action = self.agent.choose_action(obs, legal)
                hand_actions.append(action)

                if action == ACTION_STAND:
                    break
                elif action == ACTION_HIT:
                    current_hand.append(self._deal_card())
                    is_first_action = False
                elif action == ACTION_DOUBLE:
                    self.bankroll -= current_bet
                    bets[hand_idx] = current_bet * 2
                    current_bet = bets[hand_idx]
                    current_hand.append(self._deal_card())
                    break  # exactly one more card on double
                elif action == ACTION_SPLIT:
                    # Split into two hands
                    card1, card2 = current_hand[0], current_hand[1]
                    self.bankroll -= current_bet  # extra bet for second hand
                    hands[hand_idx] = [card1, self._deal_card()]
                    new_hand = [card2, self._deal_card()]
                    hands.insert(hand_idx + 1, new_hand)
                    bets.insert(hand_idx + 1, current_bet)
                    actions_taken.insert(hand_idx + 1, [])
                    # Restart current hand index (hands[hand_idx] is now new first split hand)
                    current_hand = hands[hand_idx]
                    current_bet = bets[hand_idx]
                    is_split_hand = True
                    is_first_action = True
                    continue

            hand_idx += 1

        # Dealer plays (reveal hole card)
        self.seen_cards.append(hole_card)
        while dealer_should_hit(dealer_hand, self.soft17_hit):
            dealer_hand.append(self._deal_card())

        # Compute payouts
        payouts: list[float] = []
        for h, b in zip(hands, bets):
            p = compute_payout(h, dealer_hand, b, self.blackjack_pays)
            payouts.append(p)
            self.bankroll += b + p  # return bet + net payout (p=0 on push, p=b on win, p=-b on loss)

        rec = RoundRecord(
            round_num=self.round_num,
            player_hands=hands,
            dealer_hand=dealer_hand[:],
            bets=bets,
            payouts=payouts,
            actions_taken=actions_taken,
            bankroll_after=self.bankroll,
        )
        self.round_records.append(rec)
