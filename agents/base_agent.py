"""Base class for all Blackjack agents."""
from __future__ import annotations

from abc import ABC, abstractmethod

from engine.deck import Card
from engine.game import ObservableState


class BaseAgent(ABC):
    """Abstract base for all Blackjack agents."""

    def __init__(self, player_id: int = 0) -> None:
        self.player_id = player_id
        self.last_reason: str = ""

    @abstractmethod
    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        """Return one of the legal action strings."""
        ...

    def name(self) -> str:
        return self.__class__.__name__

    def reset(self) -> None:
        """Called between sessions."""
        self.last_reason = ""
