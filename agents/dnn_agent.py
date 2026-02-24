"""DNN agent using a trained BriscasMLP-style network for Blackjack action prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from engine.deck import Card, ALL_CARDS, CARD_INDEX, hand_value, is_soft
from engine.game import ObservableState
from engine.rules import ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT, ALL_ACTIONS

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

ACTION_LIST = [ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT]
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_LIST)}
ACTION_DIM = len(ACTION_LIST)   # 4

# State vector layout (total = 128 dimensions):
# [0:52]     Player hand bitmask (card present in hand)
# [52:104]   Seen cards bitmask (all cards visible so far)
# [104:108]  Dealer upcard value one-hot (2-10=idx 0-8, A=idx 9, then 10=idx8 maps to same)
#            Actually: encode dealer upcard as 52-card one-hot → 52 dims at [52:104] for seen
#            Simpler: dealer upcard as index in ALL_CARDS → one-hot [104:156]
# [104:156]  Dealer upcard one-hot (52 dims)
# [156]      Player hand value / 21  (1 dim)
# [157]      Is soft hand flag       (1 dim)
# [158]      Is first action flag    (1 dim)
# [159]      Deck cards remaining / 312  (1 dim for 6-deck shoe)
# [160:164]  Bankroll / 1000, current_bet / 100, round_num / 100, is_split_hand (4 dims)
# [164:192]  Reserved zeros
# Total: 192 dims

STATE_DIM = 192


class StateEncoder:
    """Encode ObservableState into a fixed-size float32 numpy vector."""

    @classmethod
    def encode(cls, state: ObservableState) -> np.ndarray:
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        off = 0

        # Player hand bitmask (52 dims)
        for card in state.player_hand:
            idx = CARD_INDEX.get(card)
            if idx is not None:
                vec[off + idx] = 1.0
        off += 52

        # Seen cards bitmask (52 dims)
        for card in state.seen_cards:
            idx = CARD_INDEX.get(card)
            if idx is not None:
                vec[off + idx] = 1.0
        off += 52

        # Dealer upcard one-hot (52 dims)
        idx = CARD_INDEX.get(state.dealer_upcard)
        if idx is not None:
            vec[off + idx] = 1.0
        off += 52

        # Scalar features (4 dims)
        vec[off] = hand_value(state.player_hand) / 21.0
        vec[off + 1] = 1.0 if is_soft(state.player_hand) else 0.0
        vec[off + 2] = 1.0 if state.is_first_action else 0.0
        vec[off + 3] = state.deck_cards_remaining / 312.0
        off += 4

        # Context (4 dims)
        vec[off] = min(state.bankroll, 10000.0) / 10000.0
        vec[off + 1] = min(state.current_bet, 500.0) / 500.0
        vec[off + 2] = min(state.round_num, 200.0) / 200.0
        vec[off + 3] = 1.0 if state.is_split_hand else 0.0
        off += 4

        # Reserved: off = 164, remaining = 28 → total 192
        assert off == 164, f"Encoding offset {off} != 164"

        return vec

    @classmethod
    def action_mask(cls, legal_actions: list[str]) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=bool)
        for a in legal_actions:
            if a in ACTION_INDEX:
                mask[ACTION_INDEX[a]] = True
        return mask

    @classmethod
    def decode_action(cls, idx: int) -> str:
        return ACTION_LIST[idx]


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class BlackjackMLP(nn.Module):
    """
    MLP for Blackjack action prediction.
    Input: 192-dim state vector
    Output: 4 logits (hit / stand / double / split)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DNN Agent
# ---------------------------------------------------------------------------

class DNNAgent(BaseAgent):
    """
    Agent using a trained BlackjackMLP to predict actions.
    Illegal actions are masked to -inf before argmax.
    """

    def __init__(
        self,
        player_id: int = 0,
        model_path: str | Path = "models/blackjack_mlp.pt",
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> None:
        super().__init__(player_id)
        self.device = torch.device(device)
        self.temperature = temperature
        self.encoder = StateEncoder()

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dim = checkpoint.get("state_dim", STATE_DIM)
        action_dim = checkpoint.get("action_dim", ACTION_DIM)

        self.model = BlackjackMLP(state_dim=state_dim, action_dim=action_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def name(self) -> str:
        return "DNN"

    def choose_action(
        self,
        state: ObservableState,
        legal_actions: list[str],
    ) -> str:
        state_vec = self.encoder.encode(state)
        mask = self.encoder.action_mask(legal_actions)

        x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).squeeze(0).cpu().numpy()

        logits[~mask] = -1e9
        logits = logits / max(self.temperature, 1e-8)

        action_idx = int(np.argmax(logits))
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        confidence = int(100 * probs[action_idx])
        action = self.encoder.decode_action(action_idx)
        self.last_reason = f"DNN: '{action}' predicted with {confidence}% confidence"
        return action
