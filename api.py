"""
FastAPI backend for the Blackjack AI Tournament.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health      — liveness check
    GET  /agents      — list of available agent names
    POST /run         — run a tournament session, returns full JSON result
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.deck import Card
from engine.multi_game import (
    AgentLeaderboardEntry,
    AgentRoundResult,
    AgentRoundStep,
    MultiAgentGame,
    MultiRoundRecord,
    MultiSessionResult,
)

app = FastAPI(title="Blackjack AI Tournament API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

AVAILABLE_AGENTS = [
    "Random",
    "Heuristic(basic)",
    "Heuristic(aggressive)",
    "MCTS",
    "DNN",
]


class RunRequest(BaseModel):
    agents: List[str] = Field(default=["Random", "Heuristic(basic)"])
    num_rounds: int = Field(default=20, ge=1, le=500)
    base_bet: float = Field(default=10.0, gt=0)
    starting_bankroll: float = Field(default=1000.0, gt=0)
    seed: Optional[int] = None
    mcts_sims: int = Field(default=200, ge=10, le=2000)
    dnn_model_path: str = "models/blackjack_mlp.pt"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agents(req: RunRequest) -> list:
    agents = []
    for name in req.agents:
        n = name.lower()
        if n == "random":
            from agents.random_agent import RandomAgent
            agents.append(RandomAgent())
        elif n in ("heuristic(basic)", "heuristic basic", "heuristic - basic"):
            from agents.heuristic_agent import HeuristicAgent
            agents.append(HeuristicAgent(mode="basic"))
        elif n in ("heuristic(aggressive)", "heuristic aggressive", "heuristic - aggressive"):
            from agents.heuristic_agent import HeuristicAgent
            agents.append(HeuristicAgent(mode="aggressive"))
        elif n == "mcts":
            from agents.mcts_agent import MCTSAgent
            agents.append(MCTSAgent(n_simulations=req.mcts_sims, n_determinizations=10))
        elif n == "dnn":
            model_path = req.dnn_model_path
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"DNN model not found at '{model_path}'. "
                           "Train the model first or deselect DNN.",
                )
            from agents.dnn_agent import DNNAgent
            agents.append(DNNAgent(model_path=model_path))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent: '{name}'")
    return agents


# ---------------------------------------------------------------------------
# Serializer — converts dataclasses + Card NamedTuples → plain dicts/lists
# ---------------------------------------------------------------------------

def _card(c: Any) -> str:
    if isinstance(c, Card):
        return str(c)
    return str(c)


def _serialize_step(s: AgentRoundStep) -> dict:
    return {
        "agent_name":    s.agent_name,
        "agent_index":   s.agent_index,
        "player_hand":   [_card(c) for c in s.player_hand],
        "dealer_upcard": _card(s.dealer_upcard),
        "legal_actions": s.legal_actions,
        "action_taken":  s.action_taken,
        "reason":        s.reason,
        "hand_value":    s.hand_value,
        "is_split_hand": s.is_split_hand,
        "hand_index":    s.hand_index,
    }


def _serialize_agent_result(ar: AgentRoundResult) -> dict:
    return {
        "agent_name":    ar.agent_name,
        "agent_index":   ar.agent_index,
        "player_hands":  [[_card(c) for c in h] for h in ar.player_hands],
        "bets":          ar.bets,
        "payouts":       ar.payouts,
        "actions_taken": ar.actions_taken,
        "bankroll_after": ar.bankroll_after,
        "net_payout":    ar.net_payout,
        "steps":         [_serialize_step(s) for s in ar.steps],
    }


def _serialize_round(rec: MultiRoundRecord) -> dict:
    return {
        "round_num":     rec.round_num,
        "dealer_hand":   [_card(c) for c in rec.dealer_hand],
        "dealer_upcard": _card(rec.dealer_upcard),
        "agent_results": [_serialize_agent_result(ar) for ar in rec.agent_results],
    }


def _serialize_entry(e: AgentLeaderboardEntry) -> dict:
    return {
        "name":               e.name,
        "agent_index":        e.agent_index,
        "starting_bankroll":  e.starting_bankroll,
        "final_bankroll":     e.final_bankroll,
        "wins":               e.wins,
        "losses":             e.losses,
        "ties":               e.ties,
        "blackjacks":         e.blackjacks,
        "total_payout":       e.total_payout,
        "points":             e.points,
        "win_rate":           e.win_rate,
        "net_profit":         e.net_profit,
        "rounds_played":      e.rounds_played,
    }


def _serialize_result(result: MultiSessionResult) -> dict:
    return {
        "rounds_played":      result.rounds_played,
        "starting_bankroll":  result.starting_bankroll,
        "rounds":             [_serialize_round(r) for r in result.rounds],
        "leaderboard":        [_serialize_entry(e) for e in result.leaderboard],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> dict:
    return {"agents": AVAILABLE_AGENTS}


@app.post("/run")
def run_session(req: RunRequest) -> dict:
    if not req.agents:
        raise HTTPException(status_code=400, detail="Select at least one agent.")
    if len(req.agents) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 agents per session.")

    agents = _build_agents(req)

    game = MultiAgentGame(
        agents=agents,
        num_rounds=req.num_rounds,
        starting_bankroll=req.starting_bankroll,
        base_bet=req.base_bet,
        seed=req.seed,
    )
    result = game.run()
    return _serialize_result(result)
