# Product Requirements Document
## Agentes Inteligentes para el Juego de Blackjack
### Course: ICOM/CIIC 5015 — Artificial Intelligence

---

## 1. Project Overview

This project implements and compares four AI agent paradigms applied to the casino card game **Blackjack**. The goal is to analyze how different levels of computational reasoning — from random play to deep learning — affect performance in a partially-observable, single-player environment against a fixed dealer strategy.

---

## 2. Game Description: Blackjack

### Deck
- Standard 52-card deck (6-deck shoe used in casino play)
- Suits: Hearts, Diamonds, Clubs, Spades
- Ranks: 2–10, Jack, Queen, King, Ace

### Card Values
| Card | Value |
|------|-------|
| 2–10 | Face value |
| Jack, Queen, King | 10 |
| Ace | 1 or 11 (optimal) |

### Rules
- Dealer follows fixed strategy: hit on ≤ 16, hit on soft 17, stand on 17+
- Natural Blackjack (A + 10-value on first two cards) pays **3:2**
- Player may **Double Down** on any initial two cards
- Player may **Split** identical-rank pairs
- Shoe reshuffled when below 25% remaining
- No insurance, no surrender

### Player Actions
| Action | Description |
|--------|-------------|
| HIT | Draw one more card |
| STAND | Keep current hand |
| DOUBLE | Double bet, draw exactly one card |
| SPLIT | Split identical pair into two hands |

---

## 3. Objectives

1. Implement a complete Blackjack engine with accurate rule enforcement
2. Design and implement 4 agent types with increasing sophistication
3. Generate training datasets via expert simulation
4. Train a DNN agent via imitation learning from basic-strategy demonstrations
5. Evaluate all agents statistically over large sample sizes
6. Provide an interactive Pygame visualizer for human observation of agent sessions

---

## 4. Agent Specifications

### 4.1 Random Agent (Baseline)
- **Strategy:** Selects uniformly at random from legal actions
- **Purpose:** Statistical baseline; establishes the floor for performance
- **Complexity:** O(1) per decision

### 4.2 Heuristic Agent
Implements two modes:

**Basic Mode (Basic Strategy)**
- Follows mathematically optimal casino basic strategy for 6-deck, dealer-hits-soft-17
- Split Aces and 8s always; never split 5s or 10s
- Double on hard 10/11; soft hands per strategy table
- Stand on hard 17+; hit below threshold per dealer upcard

**Aggressive Mode**
- Hit until reaching 17+
- Always double on hard 10 or 11
- Always split Aces and 8s
- Ignores dealer upcard nuance

### 4.3 MCTS Agent (Monte Carlo Tree Search)
Addresses **imperfect information** (hidden dealer hole card) via determinization:

1. Sample `N` plausible hole cards consistent with visible information
2. Run MCTS iterations on each determinized world using UCB1 selection
3. Aggregate visit counts across all worlds
4. Select action with maximum total visits

**Parameters:**
- `n_simulations`: total MCTS iterations (default: 200)
- `n_determinizations`: number of hole card samples (default: 10)
- `ucb_c`: exploration constant (default: 1.41 ≈ √2)

**Reward:** normalized net payout per bet = `payout / bet` in [-1, +1]

### 4.4 DNN Agent (Deep Neural Network)
Trained via **imitation learning** from expert (Heuristic/MCTS) demonstrations:

**State Encoding (192-dimensional float32 vector):**
| Offset | Size | Description |
|--------|------|-------------|
| 0 | 52 | Player hand bitmask |
| 52 | 52 | Seen cards bitmask |
| 104 | 52 | Dealer upcard one-hot |
| 156 | 1 | Player hand value / 21 |
| 157 | 1 | Is soft hand flag |
| 158 | 1 | Is first action flag |
| 159 | 1 | Deck cards remaining / 312 |
| 160 | 4 | Context: bankroll/10000, bet/500, round/100, is_split |
| 164 | 28 | Reserved zeros |

**Architecture (BlackjackMLP):**
- Input: 192 → Hidden: [256 → BN → ReLU → Dropout] × 3 → Output: 4 logits
- Actions: HIT / STAND / DOUBLE / SPLIT
- Illegal actions masked to −∞ before argmax
- Trained with CrossEntropyLoss + Adam + ReduceLROnPlateau scheduler

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Visualizer                           │
│   Main Menu → Loading → Game Display → Results Screen      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      Game Engine                            │
│   engine/deck.py  ·  engine/rules.py  ·  engine/game.py   │
└──────┬──────────────────────────────────────────────┬───────┘
       │                                              │
┌──────▼──────┐                          ┌───────────▼───────┐
│   Agents    │                          │   Training /      │
│ base_agent  │                          │   Evaluation      │
│ random      │                          │                   │
│ heuristic   │                          │ dataset_generator │
│ mcts        │                          │ train_dnn         │
│ dnn         │                          │ simulate          │
└─────────────┘                          └───────────────────┘
```

### File Structure
```
blackjack/
├── engine/
│   ├── deck.py           # Card, Deck, SUITS, RANKS, hand_value, is_blackjack
│   ├── rules.py          # Actions, legal_actions, dealer_should_hit, compute_payout
│   └── game.py           # BlackjackGame, ObservableState, RoundRecord, SessionResult
├── agents/
│   ├── base_agent.py     # BaseAgent ABC, last_reason attribute
│   ├── random_agent.py   # RandomAgent
│   ├── heuristic_agent.py# HeuristicAgent (basic + aggressive)
│   ├── mcts_agent.py     # MCTSAgent with determinization
│   └── dnn_agent.py      # DNNAgent, BlackjackMLP, StateEncoder
├── training/
│   ├── dataset_generator.py  # Expert game recording → JSONL
│   └── train_dnn.py          # PyTorch training loop
├── evaluation/
│   ├── simulate.py       # Batch evaluation, multiprocessing
│   └── metrics.py        # Win rate, confidence intervals
├── visualizer.py         # Full Pygame visualizer + main menu
├── requirements.txt
└── PRD.md
```

---

## 6. Visualizer Features

### Main Menu
- **Agent selector:** carousel with Random / Heuristic (basic) / Heuristic (aggressive) / MCTS / DNN
- **MCTS settings:** simulations per move (20–1000)
- **DNN model path:** text field with live file-existence indicator
- **Session settings:** number of rounds, base bet, starting bankroll
- **Random seed:** reproducible sessions

### In-Game Display
- Dealer hand (upcard visible, hole card hidden until dealer plays)
- Player hand with live hand value and Blackjack indicator
- Bankroll panel with current bet and round counter
- Round payout result (WIN / LOSS / PUSH) when round ends
- Bankroll sparkline chart (right panel)
- Action bar with legal actions, chosen action (highlighted), and agent reasoning
- Controls hint bar

### Controls
| Key | Action |
|-----|--------|
| SPACE | Toggle auto-play / step mode |
| → / ← | Next / previous step |
| + / − | Increase / decrease speed |
| ENTER | Jump to end |
| R | Return to main menu |
| Q / Esc | Quit |

### Results Screen
- Animated slide-in
- Net profit display (green/red)
- Final bankroll vs starting bankroll
- Win/Loss/Tie summary and win rate
- Round-by-round breakdown (last 15 rounds)
- Final bankroll chart over all rounds
- Buttons: Play Again / Main Menu / Quit

---

## 7. Training Pipeline

### Dataset Generation
```bash
python training/dataset_generator.py --games 5000 --agent basic --output data/dataset.jsonl
```
Records `(state_vector, action_index)` pairs from expert games as JSONL.

### DNN Training
```bash
python training/train_dnn.py --dataset data/dataset.jsonl --epochs 50
```
- 80/20 train/validation split
- Adam optimizer, lr=1e-3, weight_decay=1e-4
- ReduceLROnPlateau scheduler (patience=5, factor=0.5)
- Early stopping on validation loss
- Saves best checkpoint to `models/blackjack_mlp.pt`

---

## 8. Evaluation Framework

```bash
python evaluation/simulate.py --agent heuristic --agent2 random --games 1000
```

### Metrics
- **Win rate** with Wilson score confidence interval (95%)
- **Average payout per round** with Student t-interval
- **Net profit** across sessions
- **Decision time** per action

### Expected Relative Performance
```
DNN (trained on basic strategy) ≈ Heuristic (basic) > Heuristic (aggressive) >> Random
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pygame | ≥ 2.0 | Visualizer |
| torch | ≥ 2.1 | DNN training & inference |
| numpy | ≥ 1.26 | State encoding, dataset |

Install:
```bash
pip install -r requirements.txt
```

---

## 10. Running the Project

```bash
# Launch the full visualizer (recommended entry point)
python visualizer.py

# Generate expert training data
python training/dataset_generator.py --games 5000

# Train the DNN
python training/train_dnn.py --dataset data/dataset.jsonl

# Run batch evaluation (single agent)
python evaluation/simulate.py --agent heuristic --games 500 --rounds 100

# Compare two agents
python evaluation/simulate.py --agent mcts --agent2 random --games 500
```

---

## 11. Academic Context

- **Course:** ICOM/CIIC 5015 — Artificial Intelligence
- **Topics covered:** Game AI, search algorithms (MCTS), machine learning (imitation learning), statistical evaluation
- **Key concepts:** Imperfect information games, determinization, UCB1, cross-entropy imitation, Wilson confidence intervals
