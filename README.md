# Blackjack AI Tournament

**Course:** ICOM/CIIC 5015 — Artificial Intelligence
**Platform:** Python 3.9+ · Pygame · PyTorch (optional)

A multi-agent Blackjack simulator where four AI paradigms compete **head-to-head against the same dealer** every round. Watch each agent make decisions in real time, see the reasoning behind every action, and compare performance on a point-based leaderboard when the session ends.

---

## Features

| Feature | Description |
|---|---|
| **Multi-agent tournament** | Up to 5 agents play every round against the same dealer hand |
| **Live decision display** | Each agent's cards, chosen action, and reasoning are shown per frame |
| **Step-by-step replay** | Pause, step forward/back, or fast-forward through the session |
| **Post-game leaderboard** | Point-based ranking (WIN=3 pts, TIE=1 pt, BLACKJACK bonus=2 pts) |
| **Bankroll history chart** | Overlaid sparkline per agent across all rounds |
| **Configurable agents** | Toggle each agent on/off from the main menu |

---

## Agent Types

### 1. Random Agent
Picks uniformly at random from all legal actions (Hit / Stand / Double / Split). Provides the statistical performance floor.

### 2. Heuristic — Basic Strategy
Follows the mathematically optimal casino basic strategy table for a 6-deck shoe with dealer-hits-soft-17. Every decision is deterministic given (player total, dealer upcard).

### 3. Heuristic — Aggressive
Always hits until reaching 17+. Doubles on hard 10 or 11. Always splits Aces and 8s. Ignores dealer upcard nuance — higher variance than basic strategy.

### 4. MCTS Agent
Uses **Monte Carlo Tree Search with determinization** to handle the hidden hole card:
1. Sample N plausible hole cards consistent with visible information.
2. Run UCB1-guided MCTS iterations on each sampled world.
3. Aggregate visit counts and select the action with the most visits.

Configurable simulations (20–1000) from the main menu.

### 5. DNN Agent *(requires trained model)*
A `BlackjackMLP` trained via **imitation learning** on expert (basic-strategy) demonstrations. Encodes a 192-dimensional state vector (hand bitmask, seen cards, upcard one-hot, normalized scalars) and outputs a masked softmax over {Hit, Stand, Double, Split}.

---

## Project Structure

```
blackjack/
├── engine/
│   ├── deck.py           # Card, Deck, hand_value, is_blackjack, is_soft
│   ├── rules.py          # Actions, legal_actions, dealer logic, compute_payout
│   ├── game.py           # Single-agent BlackjackGame, ObservableState, RoundRecord
│   └── multi_game.py     # MultiAgentGame — all agents vs same dealer per round
├── agents/
│   ├── base_agent.py     # BaseAgent ABC (last_reason attribute)
│   ├── random_agent.py   # RandomAgent
│   ├── heuristic_agent.py# HeuristicAgent (basic / aggressive modes)
│   ├── mcts_agent.py     # MCTSAgent with determinization + UCB1
│   └── dnn_agent.py      # DNNAgent + BlackjackMLP + StateEncoder
├── training/
│   ├── dataset_generator.py  # Expert demonstration → JSONL dataset
│   └── train_dnn.py          # PyTorch imitation-learning training loop
├── evaluation/
│   ├── simulate.py       # Batch evaluation, multi-agent comparison
│   └── metrics.py        # Win rate, Wilson CI, t-interval
├── visualizer.py         # Main Pygame application (multi-agent tournament UI)
├── requirements.txt
├── PRD.md                # Full product requirements document
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the tournament visualizer

```bash
python visualizer.py
```

### Main Menu Controls

- **Checkboxes** — toggle which agents participate
- **Number of Rounds** — how many rounds each agent plays
- **Base Bet / Starting Bankroll** — financial settings (same for all agents)
- **MCTS Simulations** — tree search budget per decision (higher = smarter, slower)
- **DNN Model Path** — path to a trained `.pt` checkpoint
- **Random Seed** — reproducible sessions

### In-Game Controls

| Key | Action |
|-----|--------|
| `SPACE` | Toggle auto-play / pause |
| `→` / `←` | Step forward / back one decision frame |
| `+` / `-` | Increase / decrease playback speed |
| `ENTER` | Jump to end of session |
| `R` | Return to main menu |
| `Q` / `Esc` | Quit |

---

## Leaderboard Point System

| Outcome | Points |
|---------|--------|
| Win | **3** |
| Tie (Push) | **1** |
| Loss | **0** |
| Natural Blackjack bonus | **+2** (added on top of win points) |

Agents are ranked by total points. Ties are broken by win rate, then net profit.

---

## Training the DNN Agent

### Step 1 — Generate expert demonstrations

```bash
python training/dataset_generator.py --games 5000 --agent basic --output data/dataset.jsonl
```

### Step 2 — Train the network

```bash
python training/train_dnn.py --dataset data/dataset.jsonl --epochs 50
```

The best model checkpoint is saved to `models/blackjack_mlp.pt`.
Set this path in the main menu's *DNN Model Path* field.

**Architecture:** 192-input → [256 → BN → ReLU → Dropout(0.3)] × 3 → 4 logits
**Training:** CrossEntropyLoss · Adam (lr=1e-3) · ReduceLROnPlateau · early stopping

---

## Batch Evaluation (CLI)

```bash
# Evaluate one agent over 1000 rounds
python evaluation/simulate.py --agent heuristic --games 500 --rounds 100

# Compare two agents
python evaluation/simulate.py --agent mcts --agent2 random --games 500
```

Outputs win rate with Wilson 95% confidence interval and average profit per round.

---

## Game Rules

- **Shoe:** 6 decks, reshuffled below 25% remaining
- **Dealer:** hits on soft 17, stands on hard 17+
- **Natural Blackjack:** pays 3:2
- **Double Down:** allowed on any initial two cards (not after split)
- **Split:** pairs only, no re-split, no double after split
- **No** insurance, **no** surrender

---

## Expected Performance Ranking

```
DNN (trained on basic) ≈ Heuristic (basic) > Heuristic (aggressive) >> Random
```

MCTS performance scales with simulation budget; at 200+ simulations it typically matches or slightly exceeds basic strategy.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame` | ≥ 2.0 | Tournament visualizer |
| `torch` | ≥ 2.1 | DNN training & inference |
| `numpy` | ≥ 1.26 | State encoding, dataset |

```bash
pip install pygame torch numpy
```

---

## Academic Context

- **Course:** ICOM/CIIC 5015 — Artificial Intelligence
- **Topics:** Game AI, MCTS, UCB1, imitation learning, statistical evaluation, imperfect-information games
- **Key concepts:** Determinization, cross-entropy imitation, Wilson confidence intervals, soft-17 rule
