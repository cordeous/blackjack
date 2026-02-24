"""
Generate expert Blackjack training data by recording agent decisions.

Usage:
    python training/dataset_generator.py --games 5000 --agent basic --output data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

# Allow running as: python training/dataset_generator.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engine.deck import Deck
from engine.game import BlackjackGame, ObservableState
from engine.rules import ALL_ACTIONS
from agents.base_agent import BaseAgent
from agents.dnn_agent import StateEncoder, ACTION_INDEX


class RecordingWrapper(BaseAgent):
    """Wraps any agent, recording (state_vector, action_index) pairs."""

    def __init__(self, inner: BaseAgent) -> None:
        super().__init__(inner.player_id)
        self.inner = inner
        self.records: list[dict] = []

    def name(self) -> str:
        return f"Recording({self.inner.name()})"

    def reset(self) -> None:
        self.inner.reset()
        self.last_reason = ""

    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        action = self.inner.choose_action(state, legal_actions)
        self.last_reason = self.inner.last_reason

        state_vec = StateEncoder.encode(state)
        action_idx = ACTION_INDEX.get(action, 0)

        self.records.append({
            "state": state_vec.tolist(),
            "action": action_idx,
            "legal": [ACTION_INDEX[a] for a in legal_actions if a in ACTION_INDEX],
        })
        return action


def _build_expert(agent_type: str, seed: int | None) -> BaseAgent:
    if agent_type == "basic":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="basic")
    elif agent_type == "aggressive":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="aggressive")
    elif agent_type == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(n_simulations=100, n_determinizations=5, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def generate_records(
    game_id: int,
    num_rounds: int,
    agent_type: str,
    seed: int | None,
) -> list[dict]:
    expert = _build_expert(agent_type, seed=(seed + game_id) if seed is not None else None)
    wrapper = RecordingWrapper(expert)
    game = BlackjackGame(
        agent=wrapper,
        num_rounds=num_rounds,
        seed=(seed + game_id) if seed is not None else None,
    )
    game.run()
    return wrapper.records


def generate_dataset(
    num_games: int,
    num_rounds_per_game: int,
    agent_type: str,
    output_path: Path,
    seed: int | None = None,
    workers: int = 1,
) -> int:
    """Generate dataset and write to JSONL. Returns total record count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args_list = [
        (i, num_rounds_per_game, agent_type, seed)
        for i in range(num_games)
    ]

    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        if workers > 1:
            with mp.Pool(workers) as pool:
                for records in pool.starmap(generate_records, args_list):
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                        total += 1
        else:
            for args in args_list:
                records = generate_records(*args)
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
                    total += 1
                if (args[0] + 1) % 100 == 0:
                    print(f"  [{args[0]+1}/{num_games}] {total} records so far")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Blackjack expert training data")
    parser.add_argument("--games", type=int, default=5000, help="Number of game sessions")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per session")
    parser.add_argument("--agent", type=str, default="basic",
                        choices=["basic", "aggressive", "mcts"],
                        help="Expert agent type")
    parser.add_argument("--output", type=str, default="data/dataset.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    print(f"Generating {args.games} sessions × {args.rounds} rounds with {args.agent} expert...")
    total = generate_dataset(
        num_games=args.games,
        num_rounds_per_game=args.rounds,
        agent_type=args.agent,
        output_path=Path(args.output),
        seed=args.seed,
        workers=args.workers,
    )
    print(f"Done. {total} records written to {args.output}")


if __name__ == "__main__":
    main()
