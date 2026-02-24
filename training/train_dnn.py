"""
Train a BlackjackMLP on expert imitation data.

Usage:
    python training/train_dnn.py --dataset data/dataset.jsonl --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from agents.dnn_agent import BlackjackMLP, STATE_DIM, ACTION_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BlackjackDataset(Dataset):
    """Memory-efficient JSONL dataset with line-offset indexing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offsets: list[int] = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self._offsets.append(offset)
                offset += len(line)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.path, "rb") as f:
            f.seek(self._offsets[idx])
            record = json.loads(f.readline())
        state = torch.tensor(record["state"], dtype=torch.float32)
        action = torch.tensor(record["action"], dtype=torch.long)
        return state, action


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataset_path: Path,
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    early_stop_patience: int = 10,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    dataset = BlackjackDataset(dataset_path)
    print(f"Dataset: {len(dataset)} samples")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BlackjackMLP(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(states)
            correct += (logits.argmax(1) == actions).sum().item()
            total += len(states)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
                loss = criterion(logits, actions)
                val_loss += loss.item() * len(states)
                val_correct += (logits.argmax(1) == actions).sum().item()
                val_total += len(states)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
            }, output_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BlackjackMLP via imitation learning")
    parser.add_argument("--dataset", type=str, default="data/dataset.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="models/blackjack_mlp.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
