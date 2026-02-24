"""
Batch evaluation of Blackjack agents.

Usage:
    python evaluation/simulate.py --agent basic --games 1000 --rounds 100
    python evaluation/simulate.py --agent mcts --agent2 basic --games 500
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent
from engine.game import BlackjackGame, ObservableState
from evaluation.metrics import AgentStats, EvaluationReport


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(
    agent_type: str,
    player_id: int = 0,
    mode: str = "basic",
    n_simulations: int = 200,
    n_determinizations: int = 10,
    model_path: str | None = None,
    seed: int | None = None,
) -> BaseAgent:
    if agent_type == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(player_id=player_id, seed=seed)
    elif agent_type == "heuristic":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(player_id=player_id, mode=mode)
    elif agent_type == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(
            player_id=player_id,
            n_simulations=n_simulations,
            n_determinizations=n_determinizations,
            seed=seed,
        )
    elif agent_type == "dnn":
        from agents.dnn_agent import DNNAgent
        path = model_path or "models/blackjack_mlp.pt"
        return DNNAgent(player_id=player_id, model_path=path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Time-instrumented wrapper
# ---------------------------------------------------------------------------

class TimedWrapper(BaseAgent):
    """Records per-decision timing."""

    def __init__(self, inner: BaseAgent, stats: AgentStats) -> None:
        super().__init__(inner.player_id)
        self.inner = inner
        self.stats = stats

    def name(self) -> str:
        return self.inner.name()

    def reset(self) -> None:
        self.inner.reset()

    def choose_action(self, state: ObservableState, legal_actions: list[str]) -> str:
        t0 = time.perf_counter()
        action = self.inner.choose_action(state, legal_actions)
        self.last_reason = self.inner.last_reason
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.decision_times_ms.append(elapsed_ms)
        return action


# ---------------------------------------------------------------------------
# Session worker
# ---------------------------------------------------------------------------

def run_session(
    session_id: int,
    agent_cfg: dict,
    num_rounds: int,
    starting_bankroll: float,
    base_bet: float,
    seed: int | None,
) -> dict:
    """Picklable worker: runs one session and returns outcome dict."""
    s_seed = (seed + session_id) if seed is not None else None
    agent = build_agent(seed=s_seed, **agent_cfg)
    game = BlackjackGame(
        agent=agent,
        num_rounds=num_rounds,
        starting_bankroll=starting_bankroll,
        base_bet=base_bet,
        seed=s_seed,
    )
    result = game.run()

    wins = losses = ties = 0
    payouts: list[float] = []
    for rec in result.rounds:
        net = sum(rec.payouts)
        payouts.append(net)
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        else:
            ties += 1

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "payouts": payouts,
        "final_bankroll": result.final_bankroll,
    }


def run_evaluation(
    agent_cfg: dict,
    num_sessions: int,
    num_rounds: int,
    starting_bankroll: float = 1000.0,
    base_bet: float = 10.0,
    seed: int | None = None,
    workers: int = 1,
) -> AgentStats:
    agent_name = build_agent(**agent_cfg).name()
    stats = AgentStats(agent_name=agent_name)

    args_list = [
        (i, agent_cfg, num_rounds, starting_bankroll, base_bet, seed)
        for i in range(num_sessions)
    ]

    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.starmap(run_session, args_list)
    else:
        results = [run_session(*a) for a in args_list]

    for r in results:
        stats.wins += r["wins"]
        stats.losses += r["losses"]
        stats.ties += r["ties"]
        stats.total_payout.extend(r["payouts"])
        stats.bankrolls.append(r["final_bankroll"])

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Blackjack agents")
    parser.add_argument("--agent", type=str, default="heuristic",
                        choices=["random", "heuristic", "mcts", "dnn"])
    parser.add_argument("--agent-mode", type=str, default="basic",
                        choices=["basic", "aggressive"])
    parser.add_argument("--agent-sims", type=int, default=200)
    parser.add_argument("--agent-model", type=str, default=None)
    parser.add_argument("--agent2", type=str, default=None,
                        choices=["random", "heuristic", "mcts", "dnn"])
    parser.add_argument("--agent2-mode", type=str, default="basic")
    parser.add_argument("--agent2-sims", type=int, default=200)
    parser.add_argument("--agent2-model", type=str, default=None)
    parser.add_argument("--games", type=int, default=500, help="Number of sessions")
    parser.add_argument("--rounds", type=int, default=100, help="Rounds per session")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--bet", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    all_stats: list[AgentStats] = []

    cfg1 = {
        "agent_type": args.agent,
        "mode": args.agent_mode,
        "n_simulations": args.agent_sims,
        "model_path": args.agent_model,
    }
    print(f"Evaluating {args.agent} ({args.agent_mode}) over {args.games} sessions...")
    s1 = run_evaluation(
        agent_cfg=cfg1,
        num_sessions=args.games,
        num_rounds=args.rounds,
        starting_bankroll=args.bankroll,
        base_bet=args.bet,
        seed=args.seed,
        workers=args.workers,
    )
    all_stats.append(s1)

    if args.agent2:
        cfg2 = {
            "agent_type": args.agent2,
            "mode": args.agent2_mode,
            "n_simulations": args.agent2_sims,
            "model_path": args.agent2_model,
        }
        print(f"Evaluating {args.agent2} ({args.agent2_mode}) over {args.games} sessions...")
        s2 = run_evaluation(
            agent_cfg=cfg2,
            num_sessions=args.games,
            num_rounds=args.rounds,
            starting_bankroll=args.bankroll,
            base_bet=args.bet,
            seed=args.seed,
            workers=args.workers,
        )
        all_stats.append(s2)

    report = EvaluationReport(
        stats=all_stats,
        num_sessions=args.games,
        rounds_per_session=args.rounds,
    )
    report.print_summary()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "num_sessions": args.games,
                "rounds_per_session": args.rounds,
                "agents": [
                    {
                        "name": s.agent_name,
                        "win_rate": s.win_rate,
                        "avg_payout": s.avg_payout,
                        "net_profit": s.net_profit(),
                        "wins": s.wins,
                        "losses": s.losses,
                        "ties": s.ties,
                    }
                    for s in all_stats
                ]
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
