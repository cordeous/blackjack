"""Random agent — selects uniformly from legal actions."""
from __future__ import annotations

import random

from agents.base_agent import BaseAgent
from engine.game import ObservableState


class RandomAgent(BaseAgent):
    """Selects a random legal action. Establishes the performance floor."""

    def __init__(self, player_id: int = 0, seed: int | None = None) -> None:
        super().__init__(player_id)
        self.rng = random.Random(seed)

    def name(self) -> str:
        return "Random"

    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        action = self.rng.choice(legal_actions)
        self.last_reason = f"random pick from {legal_actions}"
        return action
