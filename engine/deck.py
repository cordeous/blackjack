"""Standard 52-card deck for Blackjack."""
from __future__ import annotations

import random
from typing import NamedTuple

SUITS = ("hearts", "diamonds", "clubs", "spades")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

# Blackjack point values — Ace handled separately (1 or 11)
RANK_VALUES: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10,
    "A": 11,  # Ace counts as 11 by default; bust logic reduces to 1
}


class Card(NamedTuple):
    suit: str
    rank: str

    def value(self) -> int:
        return RANK_VALUES[self.rank]

    def is_ace(self) -> bool:
        return self.rank == "A"

    def __str__(self) -> str:
        suit_symbols = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
        return f"{self.rank}{suit_symbols[self.suit]}"

    def __repr__(self) -> str:
        return str(self)


# All 52 cards in canonical order
ALL_CARDS: list[Card] = [
    Card(suit, rank) for suit in SUITS for rank in RANKS
]

# Card index for neural network encoding
CARD_INDEX: dict[Card, int] = {card: i for i, card in enumerate(ALL_CARDS)}
INDEX_CARD: dict[int, Card] = {i: card for card, i in CARD_INDEX.items()}


def hand_value(hand: list[Card]) -> int:
    """
    Compute optimal Blackjack hand value.
    Aces count as 11, reduced to 1 to avoid bust.
    """
    total = sum(c.value() for c in hand)
    aces = sum(1 for c in hand if c.is_ace())
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def is_bust(hand: list[Card]) -> bool:
    return hand_value(hand) > 21


def is_blackjack(hand: list[Card]) -> bool:
    """Natural blackjack: exactly 2 cards totalling 21."""
    return len(hand) == 2 and hand_value(hand) == 21


def is_soft(hand: list[Card]) -> bool:
    """True if hand contains an Ace counted as 11."""
    total = sum(c.value() for c in hand)
    aces = sum(1 for c in hand if c.is_ace())
    # Soft if we have aces and at least one is counted as 11
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return aces > 0  # at least one ace still counts as 11


class Deck:
    """One or more standard 52-card decks shuffled together (shoe)."""

    def __init__(self, num_decks: int = 6, seed: int | None = None) -> None:
        self.num_decks = num_decks
        self.rng = random.Random(seed)
        self._cards: list[Card] = []
        self.shuffle()

    def shuffle(self) -> None:
        self._cards = ALL_CARDS * self.num_decks
        self.rng.shuffle(self._cards)

    def deal(self) -> Card:
        if not self._cards:
            self.shuffle()
        return self._cards.pop()

    def cards_remaining(self) -> int:
        return len(self._cards)

    def is_low(self, threshold: float = 0.25) -> bool:
        """True when shoe is below threshold fraction of original size."""
        return self.cards_remaining() < threshold * 52 * self.num_decks
