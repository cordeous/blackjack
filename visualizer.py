"""
Blackjack AI Visualizer — Pygame-based interactive observer.

Controls:
    SPACE       Toggle auto-play / step mode
    RIGHT/LEFT  Next / previous round
    + / -       Speed up / down
    R           Return to main menu
    Q / Esc     Quit
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pygame

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.deck import Card, hand_value, is_blackjack, is_bust
from engine.game import BlackjackGame, ObservableState, RoundRecord
from engine.rules import (
    ACTION_DOUBLE,
    ACTION_HIT,
    ACTION_SPLIT,
    ACTION_STAND,
    compute_payout,
)
from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Colors & layout constants
# ---------------------------------------------------------------------------

BG_COLOR        = (20, 80, 40)       # dark green felt
TABLE_COLOR     = (15, 65, 30)
CARD_BG         = (245, 245, 230)
CARD_BACK       = (25, 50, 120)
HEARTS_COLOR    = (200, 30, 30)
SPADES_COLOR    = (30, 30, 30)
CLUBS_COLOR     = (30, 30, 30)
DIAMONDS_COLOR  = (200, 30, 30)
GOLD            = (255, 215, 0)
SILVER          = (192, 192, 192)
BRONZE          = (205, 127, 50)
WHITE           = (255, 255, 255)
BLACK           = (0, 0, 0)
PANEL_BG        = (10, 50, 20)
PANEL_BORDER    = (80, 140, 80)
BUTTON_COLOR    = (60, 120, 60)
BUTTON_HOVER    = (80, 160, 80)
BUTTON_TEXT     = (255, 255, 255)
RED_ACCENT      = (200, 50, 50)
AMBER           = (255, 165, 0)
GREEN_OK        = (50, 200, 80)
TEXT_SECONDARY  = (180, 210, 180)
HIGHLIGHT       = (255, 240, 100)

W, H = 1280, 800
CARD_W, CARD_H = 72, 100
CARD_SMALL_W, CARD_SMALL_H = 55, 78

SUIT_SYMBOLS = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
SUIT_COLORS  = {
    "hearts": HEARTS_COLOR, "diamonds": DIAMONDS_COLOR,
    "clubs": CLUBS_COLOR, "spades": SPADES_COLOR,
}


# ---------------------------------------------------------------------------
# Snapshot / recording infrastructure
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of the game at one decision point."""
    round_num: int
    player_hand: list[Card]
    dealer_upcard: Card
    dealer_hand: list[Card] | None          # None until dealer reveals
    bankroll: float
    current_bet: float
    legal_actions: list[str]
    action_taken: str
    reason: str
    is_split_hand: bool
    hand_index: int
    payout: float | None = None             # set when round ends
    round_over: bool = False
    final_dealer_hand: list[Card] | None = None


class RecordingAgent(BaseAgent):
    """Wraps any agent and captures snapshots for replay."""

    def __init__(self, inner: BaseAgent) -> None:
        super().__init__(inner.player_id)
        self.inner = inner
        self.snapshots: list[Snapshot] = []
        self._pending: Snapshot | None = None

    def name(self) -> str:
        return self.inner.name()

    def reset(self) -> None:
        self.inner.reset()

    def choose_action(self, state: ObservableState, legal_actions: list[str]) -> str:
        action = self.inner.choose_action(state, legal_actions)
        self.last_reason = self.inner.last_reason

        snap = Snapshot(
            round_num=state.round_num,
            player_hand=list(state.player_hand),
            dealer_upcard=state.dealer_upcard,
            dealer_hand=None,
            bankroll=state.bankroll,
            current_bet=state.current_bet,
            legal_actions=list(legal_actions),
            action_taken=action,
            reason=self.last_reason,
            is_split_hand=state.is_split_hand,
            hand_index=state.split_hand_index,
        )
        self.snapshots.append(snap)
        return action


def record_session(cfg: "MenuConfig") -> tuple[list[Snapshot], list[RoundRecord]]:
    """Run a session with a RecordingAgent and build snapshot list."""
    inner = _build_agent(cfg)
    wrapper = RecordingAgent(inner)

    game = BlackjackGame(
        agent=wrapper,
        num_rounds=cfg.num_rounds,
        starting_bankroll=cfg.starting_bankroll,
        base_bet=cfg.base_bet,
        seed=cfg.seed,
    )
    result = game.run()

    # Annotate snapshots with round outcomes
    rounds_by_num: dict[int, RoundRecord] = {r.round_num: r for r in result.rounds}
    snap_by_round: dict[int, list[Snapshot]] = {}
    for s in wrapper.snapshots:
        snap_by_round.setdefault(s.round_num, []).append(s)

    for rnum, snaps in snap_by_round.items():
        rec = rounds_by_num.get(rnum)
        if rec:
            total_payout = sum(rec.payouts)
            for s in snaps:
                s.payout = total_payout
                s.final_dealer_hand = list(rec.dealer_hand)
            snaps[-1].round_over = True

    return wrapper.snapshots, result.rounds


def _build_agent(cfg: "MenuConfig") -> BaseAgent:
    atype = cfg.agent_type.lower()
    if atype == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent()
    elif atype in ("heuristic (basic)", "heuristic"):
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="basic")
    elif atype == "heuristic (aggressive)":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="aggressive")
    elif atype == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(n_simulations=cfg.mcts_sims, n_determinizations=10)
    elif atype == "dnn":
        model_path = cfg.dnn_model_path
        if not Path(model_path).exists():
            print(f"[WARNING] DNN model not found at '{model_path}'. Falling back to Heuristic (basic).")
            from agents.heuristic_agent import HeuristicAgent
            return HeuristicAgent(mode="basic")
        from agents.dnn_agent import DNNAgent
        return DNNAgent(model_path=model_path)
    else:
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="basic")


# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect: pygame.Rect, label: str, font: pygame.font.Font) -> None:
        self.rect = rect
        self.label = label
        self.font = font
        self.hovered = False

    def draw(self, surface: pygame.Surface) -> None:
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 2, border_radius=8)
        txt = self.font.render(self.label, True, BUTTON_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class Carousel:
    def __init__(
        self, rect: pygame.Rect, options: list[str], font: pygame.font.Font, label: str = ""
    ) -> None:
        self.rect = rect
        self.options = options
        self.index = 0
        self.font = font
        self.label = label
        btn_w = 30
        self.left_btn = pygame.Rect(rect.x, rect.y, btn_w, rect.height)
        self.right_btn = pygame.Rect(rect.right - btn_w, rect.y, btn_w, rect.height)

    @property
    def value(self) -> str:
        return self.options[self.index]

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=6)
        # Arrows
        for btn, char in [(self.left_btn, "◄"), (self.right_btn, "►")]:
            t = self.font.render(char, True, GOLD)
            surface.blit(t, t.get_rect(center=btn.center))
        # Label
        lbl = self.font.render(self.options[self.index], True, WHITE)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.left_btn.collidepoint(event.pos):
                self.index = (self.index - 1) % len(self.options)
                return True
            if self.right_btn.collidepoint(event.pos):
                self.index = (self.index + 1) % len(self.options)
                return True
        return False


class NumberSpin:
    def __init__(
        self, rect: pygame.Rect, value: int, min_val: int, max_val: int,
        step: int, font: pygame.font.Font, label: str = ""
    ) -> None:
        self.rect = rect
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.font = font
        self.label = label
        btn_w = 30
        self.left_btn = pygame.Rect(rect.x, rect.y, btn_w, rect.height)
        self.right_btn = pygame.Rect(rect.right - btn_w, rect.y, btn_w, rect.height)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=6)
        for btn, char in [(self.left_btn, "◄"), (self.right_btn, "►")]:
            t = self.font.render(char, True, GOLD)
            surface.blit(t, t.get_rect(center=btn.center))
        lbl = self.font.render(str(self.value), True, WHITE)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.left_btn.collidepoint(event.pos):
                self.value = max(self.min_val, self.value - self.step)
                return True
            if self.right_btn.collidepoint(event.pos):
                self.value = min(self.max_val, self.value + self.step)
                return True
        return False


class TextInput:
    def __init__(self, rect: pygame.Rect, font: pygame.font.Font, default: str = "") -> None:
        self.rect = rect
        self.font = font
        self.text = default
        self.active = False

    def draw(self, surface: pygame.Surface) -> None:
        color = GOLD if self.active else PANEL_BORDER
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, color, self.rect, 2, border_radius=4)
        display = self.text if self.text else ""
        txt = self.font.render(display, True, WHITE)
        # Clip to box
        clip = self.rect.inflate(-8, 0)
        surface.set_clip(clip)
        surface.blit(txt, (self.rect.x + 6, self.rect.centery - txt.get_height() // 2))
        surface.set_clip(None)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key not in (pygame.K_RETURN, pygame.K_TAB, pygame.K_ESCAPE):
                self.text += event.unicode
            return True
        return False


# ---------------------------------------------------------------------------
# Menu config
# ---------------------------------------------------------------------------

@dataclass
class MenuConfig:
    agent_type: str = "Heuristic (basic)"
    mcts_sims: int = 200
    dnn_model_path: str = "models/blackjack_mlp.pt"
    num_rounds: int = 20
    starting_bankroll: float = 1000.0
    base_bet: float = 10.0
    seed: int | None = None


AGENT_OPTIONS = [
    "Random",
    "Heuristic (basic)",
    "Heuristic (aggressive)",
    "MCTS",
    "DNN",
]


# ---------------------------------------------------------------------------
# Card rendering helpers
# ---------------------------------------------------------------------------

def draw_card(
    surface: pygame.Surface,
    card: Card,
    x: int,
    y: int,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    w: int = CARD_W,
    h: int = CARD_H,
    highlighted: bool = False,
) -> None:
    rect = pygame.Rect(x, y, w, h)
    color = HIGHLIGHT if highlighted else CARD_BG
    pygame.draw.rect(surface, color, rect, border_radius=7)
    pygame.draw.rect(surface, (80, 80, 80), rect, 1, border_radius=7)

    suit_sym = SUIT_SYMBOLS[card.suit]
    suit_col = SUIT_COLORS[card.suit]
    rank_txt = font_sm.render(card.rank, True, suit_col)
    suit_sm  = font_sm.render(suit_sym, True, suit_col)
    rank_lg  = font_lg.render(card.rank, True, suit_col)
    suit_lg  = font_lg.render(suit_sym, True, suit_col)

    surface.blit(rank_txt, (x + 4, y + 2))
    surface.blit(suit_sm,  (x + 4, y + 2 + rank_txt.get_height()))
    # Center large suit
    surface.blit(suit_lg, suit_lg.get_rect(center=(x + w // 2, y + h // 2)))


def draw_card_back(
    surface: pygame.Surface,
    x: int, y: int,
    w: int = CARD_W, h: int = CARD_H,
) -> None:
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, CARD_BACK, rect, border_radius=7)
    pygame.draw.rect(surface, (80, 80, 200), rect, 2, border_radius=7)
    inner = rect.inflate(-10, -10)
    pygame.draw.rect(surface, (40, 60, 150), inner, border_radius=5)


def hand_display_x(n_cards: int, center_x: int, card_w: int, gap: int = 10) -> int:
    """Left x for a centered row of n_cards."""
    total_w = n_cards * card_w + (n_cards - 1) * gap
    return center_x - total_w // 2


# ---------------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------------

class MainMenu:
    def __init__(self, screen: pygame.Surface, fonts: dict) -> None:
        self.screen = screen
        self.fonts = fonts
        self.cfg = MenuConfig()

        cx = W // 2
        row_h = 60
        start_y = 200

        self.agent_carousel = Carousel(
            pygame.Rect(cx - 200, start_y, 400, 44),
            AGENT_OPTIONS, fonts["md"], "Agent"
        )
        self.mcts_spin = NumberSpin(
            pygame.Rect(cx - 100, start_y + row_h, 200, 44),
            200, 20, 1000, 20, fonts["md"], "MCTS Simulations"
        )
        self.rounds_spin = NumberSpin(
            pygame.Rect(cx - 100, start_y + row_h * 2, 200, 44),
            20, 1, 200, 5, fonts["md"], "Rounds"
        )
        self.bet_spin = NumberSpin(
            pygame.Rect(cx - 100, start_y + row_h * 3, 200, 44),
            10, 1, 200, 5, fonts["md"], "Base Bet ($)"
        )
        self.bankroll_spin = NumberSpin(
            pygame.Rect(cx - 100, start_y + row_h * 4, 200, 44),
            1000, 100, 10000, 100, fonts["md"], "Starting Bankroll ($)"
        )
        self.seed_spin = NumberSpin(
            pygame.Rect(cx - 100, start_y + row_h * 5, 200, 44),
            42, 0, 9999, 1, fonts["md"], "Seed"
        )
        self.dnn_input = TextInput(
            pygame.Rect(cx - 200, start_y + row_h * 6, 400, 40),
            fonts["sm"], "models/blackjack_mlp.pt"
        )
        self.start_btn = Button(
            pygame.Rect(cx - 120, start_y + row_h * 7 + 10, 240, 50),
            "▶  PLAY", fonts["md"]
        )
        self.quit_btn = Button(
            pygame.Rect(cx - 60, start_y + row_h * 7 + 70, 120, 40),
            "Quit", fonts["sm"]
        )
        self._widgets = [
            self.agent_carousel, self.mcts_spin, self.rounds_spin,
            self.bet_spin, self.bankroll_spin, self.seed_spin, self.dnn_input,
        ]

    def run(self) -> MenuConfig | None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return None
                for w in self._widgets:
                    w.handle_event(event)
                if self.start_btn.handle_event(event):
                    return self._build_cfg()
                if self.quit_btn.handle_event(event):
                    return None

            self._draw()
            clock.tick(60)

    def _build_cfg(self) -> MenuConfig:
        return MenuConfig(
            agent_type=self.agent_carousel.value,
            mcts_sims=self.mcts_spin.value,
            dnn_model_path=self.dnn_input.text,
            num_rounds=self.rounds_spin.value,
            starting_bankroll=float(self.bankroll_spin.value),
            base_bet=float(self.bet_spin.value),
            seed=self.seed_spin.value,
        )

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)

        # Title
        title = self.fonts["title"].render("♠  Blackjack AI  ♥", True, GOLD)
        self.screen.blit(title, title.get_rect(centerx=W // 2, y=50))
        sub = self.fonts["sm"].render("ICOM/CIIC 5015 — Artificial Intelligence", True, TEXT_SECONDARY)
        self.screen.blit(sub, sub.get_rect(centerx=W // 2, y=120))

        cx = W // 2
        row_h = 60
        start_y = 200

        labels = [
            (start_y - 18,     "Agent Type"),
            (start_y + row_h - 18,  "MCTS Simulations"),
            (start_y + row_h * 2 - 18, "Number of Rounds"),
            (start_y + row_h * 3 - 18, "Base Bet ($)"),
            (start_y + row_h * 4 - 18, "Starting Bankroll ($)"),
            (start_y + row_h * 5 - 18, "Random Seed"),
            (start_y + row_h * 6 - 18, "DNN Model Path"),
        ]
        for y_lbl, lbl in labels:
            t = self.fonts["xs"].render(lbl, True, TEXT_SECONDARY)
            self.screen.blit(t, t.get_rect(centerx=cx, y=y_lbl))

        # DNN file indicator
        dnn_path = self.dnn_input.text
        exists = Path(dnn_path).exists()
        ind_txt = "✓ found" if exists else "✗ not found"
        ind_col = GREEN_OK if exists else RED_ACCENT
        ind = self.fonts["xs"].render(ind_txt, True, ind_col)
        self.screen.blit(ind, (cx + 210, start_y + row_h * 6 + 10))

        for w in self._widgets:
            w.draw(self.screen)
        self.start_btn.draw(self.screen)
        self.quit_btn.draw(self.screen)

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Game Visualizer
# ---------------------------------------------------------------------------

class GameVisualizer:
    def __init__(
        self,
        screen: pygame.Surface,
        fonts: dict,
        snapshots: list[Snapshot],
        rounds: list[RoundRecord],
        cfg: MenuConfig,
    ) -> None:
        self.screen = screen
        self.fonts = fonts
        self.snapshots = snapshots
        self.rounds = rounds
        self.cfg = cfg
        self.snap_idx = 0
        self.auto_play = True
        self.speed = 1.0   # seconds per step
        self._last_step = time.time()

        self.menu_btn = Button(pygame.Rect(10, 10, 110, 36), "← Menu", fonts["xs"])
        self.play_btn = Button(pygame.Rect(130, 10, 100, 36), "⏸ Pause", fonts["xs"])

    def run(self) -> str:
        """Returns 'menu' or 'quit'."""
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    if event.key == pygame.K_r:
                        return "menu"
                    if event.key == pygame.K_SPACE:
                        self.auto_play = not self.auto_play
                    if event.key == pygame.K_RIGHT:
                        self._advance()
                    if event.key == pygame.K_LEFT:
                        self.snap_idx = max(0, self.snap_idx - 1)
                    if event.key == pygame.K_RETURN:
                        self.snap_idx = len(self.snapshots) - 1
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.speed = max(0.1, self.speed / 1.5)
                    if event.key == pygame.K_MINUS:
                        self.speed = min(5.0, self.speed * 1.5)
                if self.menu_btn.handle_event(event):
                    return "menu"
                if self.play_btn.handle_event(event):
                    self.auto_play = not self.auto_play

            if self.auto_play and time.time() - self._last_step >= self.speed:
                if self.snap_idx < len(self.snapshots) - 1:
                    self._advance()
                else:
                    self.auto_play = False

            self.play_btn.label = "⏸ Pause" if self.auto_play else "▶ Play"
            self._draw()
            clock.tick(60)

    def _advance(self) -> None:
        self.snap_idx = min(len(self.snapshots) - 1, self.snap_idx + 1)
        self._last_step = time.time()

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)
        if not self.snapshots:
            return

        snap = self.snapshots[self.snap_idx]
        fonts = self.fonts

        # Header bar
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, 55))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, 55), (W, 55))

        agent_name = self.cfg.agent_type
        hdr = fonts["md"].render(f"Agent: {agent_name}", True, GOLD)
        self.screen.blit(hdr, hdr.get_rect(centerx=W // 2, centery=27))

        self.menu_btn.draw(self.screen)
        self.play_btn.draw(self.screen)

        # Speed indicator
        spd_txt = fonts["xs"].render(f"Speed: {1/self.speed:.1f}x", True, TEXT_SECONDARY)
        self.screen.blit(spd_txt, (W - 140, 18))

        # Step counter
        ctr = fonts["xs"].render(f"Step {self.snap_idx+1}/{len(self.snapshots)}", True, TEXT_SECONDARY)
        self.screen.blit(ctr, (250, 18))

        # Round info
        rnd_txt = fonts["sm"].render(f"Round {snap.round_num}", True, WHITE)
        self.screen.blit(rnd_txt, rnd_txt.get_rect(x=W - 300, centery=27))

        # --- Layout ---
        table_rect = pygame.Rect(100, 70, W - 200, H - 200)

        # DEALER section (top)
        dealer_y = 90
        dealer_label = fonts["sm"].render("DEALER", True, TEXT_SECONDARY)
        self.screen.blit(dealer_label, dealer_label.get_rect(centerx=W // 2, y=dealer_y))
        dealer_y += 28

        show_dealer_full = snap.round_over or snap.final_dealer_hand is not None
        if show_dealer_full and snap.final_dealer_hand:
            dealer_cards = snap.final_dealer_hand
        else:
            dealer_cards = [snap.dealer_upcard]  # only upcard visible

        n_d = len(dealer_cards) + (0 if show_dealer_full else 1)  # +1 for hidden
        dx_start = hand_display_x(n_d, W // 2, CARD_W)
        # Draw upcard
        draw_card(self.screen, snap.dealer_upcard, dx_start, dealer_y,
                  fonts["card_lg"], fonts["card_sm"])
        if show_dealer_full and snap.final_dealer_hand:
            for i, c in enumerate(snap.final_dealer_hand[1:], 1):
                draw_card(self.screen, c, dx_start + i * (CARD_W + 10), dealer_y,
                          fonts["card_lg"], fonts["card_sm"])
        else:
            draw_card_back(self.screen, dx_start + CARD_W + 10, dealer_y)

        # Dealer value
        if show_dealer_full and snap.final_dealer_hand:
            dval = hand_value(snap.final_dealer_hand)
            dval_txt = fonts["sm"].render(f"Value: {dval}", True,
                                           RED_ACCENT if dval > 21 else WHITE)
            self.screen.blit(dval_txt, dval_txt.get_rect(centerx=W // 2, y=dealer_y + CARD_H + 5))

        # PLAYER section (bottom)
        player_y = H - 250
        player_label = fonts["sm"].render("PLAYER", True, TEXT_SECONDARY)
        self.screen.blit(player_label, player_label.get_rect(centerx=W // 2, y=player_y))
        player_y += 28

        n_p = len(snap.player_hand)
        px_start = hand_display_x(n_p, W // 2, CARD_W)
        for i, c in enumerate(snap.player_hand):
            draw_card(self.screen, c, px_start + i * (CARD_W + 10), player_y,
                      fonts["card_lg"], fonts["card_sm"])

        pval = hand_value(snap.player_hand)
        pval_color = RED_ACCENT if pval > 21 else (GOLD if pval == 21 else WHITE)
        bj_note = "  BLACKJACK!" if is_blackjack(snap.player_hand) else ""
        pval_txt = fonts["sm"].render(f"Value: {pval}{bj_note}", True, pval_color)
        self.screen.blit(pval_txt, pval_txt.get_rect(centerx=W // 2, y=player_y + CARD_H + 5))

        # Payout result
        if snap.round_over and snap.payout is not None:
            payout = snap.payout
            if payout > 0:
                res_txt, res_col = f"+${payout:.0f}  WIN!", GREEN_OK
            elif payout < 0:
                res_txt, res_col = f"-${abs(payout):.0f}  LOSS", RED_ACCENT
            else:
                res_txt, res_col = "PUSH (tie)", AMBER
            res = fonts["lg"].render(res_txt, True, res_col)
            self.screen.blit(res, res.get_rect(centerx=W // 2, y=player_y - 50))

        # Side panel — bankroll & bet
        panel = pygame.Rect(20, 80, 190, 160)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1, border_radius=8)
        bk_txt = fonts["sm"].render("Bankroll", True, TEXT_SECONDARY)
        self.screen.blit(bk_txt, (30, 90))
        bk_val = fonts["lg"].render(f"${snap.bankroll:.0f}", True, GOLD)
        self.screen.blit(bk_val, (30, 110))
        bet_lbl = fonts["sm"].render("Bet", True, TEXT_SECONDARY)
        self.screen.blit(bet_lbl, (30, 148))
        bet_val = fonts["md"].render(f"${snap.current_bet:.0f}", True, WHITE)
        self.screen.blit(bet_val, (30, 168))
        rnd_lbl = fonts["sm"].render("Round", True, TEXT_SECONDARY)
        self.screen.blit(rnd_lbl, (30, 200))
        rnd_val = fonts["md"].render(f"{snap.round_num} / {self.cfg.num_rounds}", True, WHITE)
        self.screen.blit(rnd_val, (30, 220))

        # Right panel — bankroll history mini chart
        self._draw_bankroll_chart()

        # Action bar
        action_bar = pygame.Rect(0, H - 120, W, 120)
        pygame.draw.rect(self.screen, PANEL_BG, action_bar)
        pygame.draw.line(self.screen, PANEL_BORDER, (0, H - 120), (W, H - 120))

        # Legal actions
        actions_label = fonts["xs"].render("Legal actions:", True, TEXT_SECONDARY)
        self.screen.blit(actions_label, (20, H - 115))
        act_x = 130
        for a in snap.legal_actions:
            col = GOLD if a == snap.action_taken else WHITE
            atxt = fonts["sm"].render(f"[{a.upper()}]", True, col)
            self.screen.blit(atxt, (act_x, H - 115))
            act_x += atxt.get_width() + 12

        # Chosen action
        chosen_lbl = fonts["xs"].render("Action taken:", True, TEXT_SECONDARY)
        self.screen.blit(chosen_lbl, (20, H - 90))
        chosen_txt = fonts["lg"].render(snap.action_taken.upper(), True, GOLD)
        self.screen.blit(chosen_txt, (140, H - 95))

        # Reason
        reason_txt = fonts["xs"].render(f"· {snap.reason}", True, TEXT_SECONDARY)
        self.screen.blit(reason_txt, (20, H - 55))

        # Controls hint
        hint = fonts["xs"].render(
            "SPACE: play/pause  |  ◄►: step  |  +/-: speed  |  ENTER: end  |  R: menu  |  Q: quit",
            True, (100, 140, 100)
        )
        self.screen.blit(hint, hint.get_rect(centerx=W // 2, y=H - 25))

        pygame.display.flip()

    def _draw_bankroll_chart(self) -> None:
        """Small sparkline of bankroll over rounds on the right panel."""
        if not self.rounds:
            return
        panel = pygame.Rect(W - 210, 80, 190, 160)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1, border_radius=8)
        lbl = self.fonts["xs"].render("Bankroll History", True, TEXT_SECONDARY)
        self.screen.blit(lbl, (W - 205, 85))

        bankrolls = [r.bankroll_after for r in self.rounds]
        if len(bankrolls) < 2:
            return

        chart_rect = pygame.Rect(W - 205, 104, 180, 120)
        min_b = min(bankrolls)
        max_b = max(bankrolls)
        rng = max(max_b - min_b, 1.0)

        def to_screen(i: int, b: float) -> tuple[int, int]:
            x = chart_rect.x + int(i / (len(bankrolls) - 1) * chart_rect.width)
            y = chart_rect.bottom - int((b - min_b) / rng * chart_rect.height)
            return x, y

        pts = [to_screen(i, b) for i, b in enumerate(bankrolls)]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, GREEN_OK, False, pts, 2)

        # Starting bankroll line
        start_bk = self.cfg.starting_bankroll
        sy = chart_rect.bottom - int((start_bk - min_b) / rng * chart_rect.height)
        pygame.draw.line(self.screen, AMBER, (chart_rect.x, sy), (chart_rect.right, sy), 1)

        # Current value label
        final = bankrolls[-1]
        fc = GREEN_OK if final >= self.cfg.starting_bankroll else RED_ACCENT
        ftxt = self.fonts["xs"].render(f"${final:.0f}", True, fc)
        self.screen.blit(ftxt, (W - 205, 228))


# ---------------------------------------------------------------------------
# Results screen
# ---------------------------------------------------------------------------

class ResultsScreen:
    def __init__(
        self,
        screen: pygame.Surface,
        fonts: dict,
        rounds: list[RoundRecord],
        cfg: MenuConfig,
    ) -> None:
        self.screen = screen
        self.fonts = fonts
        self.rounds = rounds
        self.cfg = cfg
        cx = W // 2
        self.play_again_btn = Button(pygame.Rect(cx - 200, H - 130, 180, 50), "Play Again", fonts["md"])
        self.menu_btn = Button(pygame.Rect(cx - 10, H - 130, 120, 50), "Main Menu", fonts["sm"])
        self.quit_btn = Button(pygame.Rect(cx + 120, H - 130, 90, 50), "Quit", fonts["sm"])
        self._slide = 0.0  # animation progress 0→1

    def run(self) -> str:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    if event.key == pygame.K_r:
                        return "menu"
                if self.play_again_btn.handle_event(event):
                    return "play_again"
                if self.menu_btn.handle_event(event):
                    return "menu"
                if self.quit_btn.handle_event(event):
                    return "quit"

            self._slide = min(1.0, self._slide + 0.04)
            self._draw()
            clock.tick(60)

    def _draw(self) -> None:
        offset_y = int((1.0 - self._slide) * H)
        self.screen.fill(BG_COLOR)
        fonts = self.fonts
        cx = W // 2

        # Title
        t = fonts["title"].render("Session Complete", True, GOLD)
        self.screen.blit(t, t.get_rect(centerx=cx, y=30 + offset_y))

        # Summary stats
        if not self.rounds:
            return

        total_payout = sum(sum(r.payouts) for r in self.rounds)
        wins = sum(1 for r in self.rounds if sum(r.payouts) > 0)
        losses = sum(1 for r in self.rounds if sum(r.payouts) < 0)
        ties = sum(1 for r in self.rounds if sum(r.payouts) == 0)
        final_bk = self.rounds[-1].bankroll_after
        start_bk = self.cfg.starting_bankroll

        pnl_color = GREEN_OK if total_payout >= 0 else RED_ACCENT
        pnl_txt = f"Net Profit: ${total_payout:+.0f}"
        pnl = fonts["lg"].render(pnl_txt, True, pnl_color)
        self.screen.blit(pnl, pnl.get_rect(centerx=cx, y=100 + offset_y))

        bk_txt = fonts["md"].render(
            f"Final Bankroll: ${final_bk:.0f}  (started ${start_bk:.0f})", True, WHITE
        )
        self.screen.blit(bk_txt, bk_txt.get_rect(centerx=cx, y=150 + offset_y))

        wr = wins / max(len(self.rounds), 1)
        wr_txt = fonts["sm"].render(
            f"Win Rate: {wr:.1%}  |  Rounds: {len(self.rounds)}  |  W:{wins} / L:{losses} / T:{ties}",
            True, TEXT_SECONDARY
        )
        self.screen.blit(wr_txt, wr_txt.get_rect(centerx=cx, y=190 + offset_y))

        # Round-by-round breakdown (show last N)
        show_n = min(15, len(self.rounds))
        panel = pygame.Rect(cx - 350, 240 + offset_y, 700, 30 + show_n * 28)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1, border_radius=8)

        hdr = fonts["xs"].render("Round   Player Hand     Dealer Hand     Payout", True, TEXT_SECONDARY)
        self.screen.blit(hdr, (panel.x + 10, panel.y + 6))

        for i, rec in enumerate(self.rounds[-show_n:]):
            y = panel.y + 30 + i * 28
            payout = sum(rec.payouts)
            col = GREEN_OK if payout > 0 else (RED_ACCENT if payout < 0 else AMBER)

            pval = hand_value(rec.player_hands[0])
            dval = hand_value(rec.dealer_hand)
            cards_str = " ".join(str(c) for c in rec.player_hands[0])
            dealer_str = " ".join(str(c) for c in rec.dealer_hand)

            row = f"  {rec.round_num:3d}     {cards_str:<18} {dealer_str:<18} {payout:+.0f}"
            rtxt = fonts["xs"].render(row, True, col)
            self.screen.blit(rtxt, (panel.x + 5, y))

        # Bankroll chart
        self._draw_final_chart(offset_y)

        self.play_again_btn.draw(self.screen)
        self.menu_btn.draw(self.screen)
        self.quit_btn.draw(self.screen)

        pygame.display.flip()

    def _draw_final_chart(self, offset_y: int) -> None:
        if len(self.rounds) < 2:
            return
        chart_rect = pygame.Rect(W - 260, 260 + offset_y, 240, 160)
        pygame.draw.rect(self.screen, PANEL_BG, chart_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, chart_rect, 1, border_radius=8)
        lbl = self.fonts["xs"].render("Bankroll Over Time", True, TEXT_SECONDARY)
        self.screen.blit(lbl, (chart_rect.x + 6, chart_rect.y + 4))

        bankrolls = [r.bankroll_after for r in self.rounds]
        min_b = min(bankrolls)
        max_b = max(bankrolls)
        rng = max(max_b - min_b, 1.0)
        inner = chart_rect.inflate(-16, -24)
        inner.y += 18

        def tp(i: int, b: float) -> tuple[int, int]:
            x = inner.x + int(i / (len(bankrolls) - 1) * inner.width)
            y = inner.bottom - int((b - min_b) / rng * inner.height)
            return x, y

        pts = [tp(i, b) for i, b in enumerate(bankrolls)]
        pygame.draw.lines(self.screen, GREEN_OK, False, pts, 2)

        # Start line
        sy = inner.bottom - int((self.cfg.starting_bankroll - min_b) / rng * inner.height)
        sy = max(inner.y, min(inner.bottom, sy))
        pygame.draw.line(self.screen, AMBER, (inner.x, sy), (inner.right, sy), 1)


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------

def show_loading(screen: pygame.Surface, fonts: dict) -> None:
    screen.fill(BG_COLOR)
    msg = fonts["lg"].render("Simulating rounds...", True, GOLD)
    screen.blit(msg, msg.get_rect(center=(W // 2, H // 2)))
    sub = fonts["sm"].render("Please wait", True, TEXT_SECONDARY)
    screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 50)))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Blackjack AI — ICOM/CIIC 5015")

    try:
        title_font = pygame.font.SysFont("segoeui", 52, bold=True)
        lg_font    = pygame.font.SysFont("segoeui", 34, bold=True)
        md_font    = pygame.font.SysFont("segoeui", 24)
        sm_font    = pygame.font.SysFont("segoeui", 18)
        xs_font    = pygame.font.SysFont("segoeui", 14)
        card_lg    = pygame.font.SysFont("segoeui", 24, bold=True)
        card_sm    = pygame.font.SysFont("segoeui", 14)
    except Exception:
        title_font = pygame.font.Font(None, 62)
        lg_font    = pygame.font.Font(None, 42)
        md_font    = pygame.font.Font(None, 30)
        sm_font    = pygame.font.Font(None, 24)
        xs_font    = pygame.font.Font(None, 18)
        card_lg    = pygame.font.Font(None, 28)
        card_sm    = pygame.font.Font(None, 18)

    fonts = {
        "title": title_font,
        "lg": lg_font,
        "md": md_font,
        "sm": sm_font,
        "xs": xs_font,
        "card_lg": card_lg,
        "card_sm": card_sm,
    }

    while True:
        # Main menu
        menu = MainMenu(screen, fonts)
        cfg = menu.run()
        if cfg is None:
            break

        # Record session
        show_loading(screen, fonts)
        pygame.event.pump()
        snapshots, rounds = record_session(cfg)

        if not snapshots:
            continue

        # Game visualizer
        visualizer = GameVisualizer(screen, fonts, snapshots, rounds, cfg)
        result = visualizer.run()
        if result == "quit":
            break

        if result == "menu":
            continue

        # Results screen
        results = ResultsScreen(screen, fonts, rounds, cfg)
        outcome = results.run()
        if outcome == "quit":
            break
        elif outcome == "play_again":
            show_loading(screen, fonts)
            pygame.event.pump()
            snapshots, rounds = record_session(cfg)
            visualizer = GameVisualizer(screen, fonts, snapshots, rounds, cfg)
            r2 = visualizer.run()
            if r2 == "quit":
                break
            # Loop back to menu

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
