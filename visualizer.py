"""
Blackjack AI Visualizer — Multi-agent Pygame interface.

All active agents play against the same dealer each round.
Displays each agent's hand, their decision, and the reasoning behind it.
After all rounds, a point-based leaderboard is shown.

Controls (in-game):
    SPACE       Toggle auto-play / step mode
    RIGHT/LEFT  Next / previous step
    + / -       Increase / decrease speed
    ENTER       Jump to end of session
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.deck import Card, hand_value, is_blackjack, is_bust
from engine.rules import ACTION_DOUBLE, ACTION_HIT, ACTION_SPLIT, ACTION_STAND
from engine.multi_game import (
    AgentRoundResult,
    AgentRoundStep,
    MultiAgentGame,
    MultiRoundRecord,
    MultiSessionResult,
    AgentLeaderboardEntry,
)
from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Window dimensions
# ---------------------------------------------------------------------------

W, H = 1440, 900

# ---------------------------------------------------------------------------
# Color palette — rich casino green feel
# ---------------------------------------------------------------------------

BG_COLOR       = (18, 74, 38)        # deep felt green
TABLE_FELT     = (22, 88, 46)
FELT_DARK      = (12, 55, 28)
CARD_BG        = (248, 248, 235)
CARD_BACK_TOP  = (30, 55, 135)
CARD_BACK_BOT  = (20, 38, 100)
HEARTS_COL     = (210, 35, 35)
SPADES_COL     = (28, 28, 28)
CLUBS_COL      = (28, 28, 28)
DIAMONDS_COL   = (210, 35, 35)
GOLD           = (255, 215, 0)
GOLD_DARK      = (200, 165, 0)
SILVER         = (192, 192, 192)
BRONZE         = (205, 127, 50)
WHITE          = (255, 255, 255)
BLACK          = (0, 0, 0)
PANEL_BG       = (10, 48, 22)
PANEL_BG2      = (14, 60, 30)
PANEL_BORDER   = (60, 130, 70)
PANEL_BORDER_BRIGHT = (100, 200, 110)
BTN_NORM       = (55, 115, 60)
BTN_HOVER      = (75, 155, 80)
BTN_TEXT       = (255, 255, 255)
RED_ACCENT     = (210, 55, 55)
AMBER          = (255, 170, 0)
GREEN_OK       = (55, 210, 90)
TEXT_DIM       = (150, 195, 155)
TEXT_MED       = (210, 235, 210)
HIGHLIGHT      = (255, 245, 110)
SHADOW         = (0, 0, 0, 120)

# Agent colour identities (for visual distinction)
AGENT_COLORS = [
    (255, 200,  60),   # gold   — Random
    ( 80, 190, 255),   # blue   — Heuristic basic
    (255, 130,  60),   # orange — Heuristic aggressive
    (160, 100, 255),   # purple — MCTS
    ( 60, 230, 160),   # teal   — DNN
]

SUIT_SYMBOLS = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
SUIT_COLORS  = {
    "hearts": HEARTS_COL, "diamonds": DIAMONDS_COL,
    "clubs": CLUBS_COL,   "spades": SPADES_COL,
}

CARD_W, CARD_H       = 58, 82
CARD_SM_W, CARD_SM_H = 42, 60


# ---------------------------------------------------------------------------
# UI helper widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect: pygame.Rect, label: str, font: pygame.font.Font,
                 color: tuple = BTN_NORM, hover_color: tuple = BTN_HOVER) -> None:
        self.rect = rect
        self.label = label
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.hovered = False

    def draw(self, surface: pygame.Surface) -> None:
        col = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, col, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER_BRIGHT, self.rect, 2, border_radius=8)
        txt = self.font.render(self.label, True, BTN_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class Carousel:
    def __init__(self, rect: pygame.Rect, options: list[str],
                 font: pygame.font.Font) -> None:
        self.rect = rect
        self.options = options
        self.index = 0
        self.font = font
        bw = 32
        self.left_btn  = pygame.Rect(rect.x, rect.y, bw, rect.height)
        self.right_btn = pygame.Rect(rect.right - bw, rect.y, bw, rect.height)

    @property
    def value(self) -> str:
        return self.options[self.index]

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=6)
        for btn, char in [(self.left_btn, "◄"), (self.right_btn, "►")]:
            t = self.font.render(char, True, GOLD)
            surface.blit(t, t.get_rect(center=btn.center))
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
    def __init__(self, rect: pygame.Rect, value: int,
                 min_val: int, max_val: int, step: int,
                 font: pygame.font.Font) -> None:
        self.rect = rect
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.font = font
        bw = 32
        self.left_btn  = pygame.Rect(rect.x, rect.y, bw, rect.height)
        self.right_btn = pygame.Rect(rect.right - bw, rect.y, bw, rect.height)

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


class CheckBox:
    def __init__(self, rect: pygame.Rect, label: str,
                 font: pygame.font.Font, checked: bool = True) -> None:
        self.rect = rect
        self.label = label
        self.font = font
        self.checked = checked
        self.box = pygame.Rect(rect.x, rect.centery - 10, 20, 20)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.box, border_radius=3)
        pygame.draw.rect(surface, PANEL_BORDER, self.box, 1, border_radius=3)
        if self.checked:
            inner = self.box.inflate(-6, -6)
            pygame.draw.rect(surface, GREEN_OK, inner, border_radius=2)
        lbl = self.font.render(self.label, True, TEXT_MED)
        surface.blit(lbl, (self.box.right + 8, self.rect.centery - lbl.get_height() // 2))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and self.box.collidepoint(event.pos):
            self.checked = not self.checked
            return True
        return False


class TextInput:
    def __init__(self, rect: pygame.Rect, font: pygame.font.Font,
                 default: str = "") -> None:
        self.rect = rect
        self.font = font
        self.text = default
        self.active = False

    def draw(self, surface: pygame.Surface) -> None:
        color = GOLD if self.active else PANEL_BORDER
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, color, self.rect, 2, border_radius=4)
        txt = self.font.render(self.text, True, WHITE)
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
# Card rendering
# ---------------------------------------------------------------------------

def draw_card(surface: pygame.Surface, card: Card, x: int, y: int,
              font_lg: pygame.font.Font, font_sm: pygame.font.Font,
              w: int = CARD_W, h: int = CARD_H,
              highlighted: bool = False, dim: bool = False) -> None:
    rect = pygame.Rect(x, y, w, h)
    bg = HIGHLIGHT if highlighted else CARD_BG
    # Shadow
    shadow_rect = rect.move(3, 3)
    shadow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    shadow_surf.fill((0, 0, 0, 80))
    pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=7)
    surface.blit(shadow_surf, (shadow_rect.x, shadow_rect.y))

    pygame.draw.rect(surface, bg, rect, border_radius=7)
    pygame.draw.rect(surface, (100, 100, 90) if not highlighted else GOLD_DARK, rect, 1, border_radius=7)

    suit_sym = SUIT_SYMBOLS[card.suit]
    suit_col = SUIT_COLORS[card.suit]
    if dim:
        suit_col = tuple(max(0, c - 80) for c in suit_col)

    rank_txt = font_sm.render(card.rank, True, suit_col)
    suit_sm  = font_sm.render(suit_sym, True, suit_col)
    suit_lg  = font_lg.render(suit_sym, True, suit_col)

    surface.blit(rank_txt, (x + 4, y + 3))
    surface.blit(suit_sm,  (x + 4, y + 3 + rank_txt.get_height()))
    surface.blit(suit_lg, suit_lg.get_rect(center=(x + w // 2, y + h // 2)))


def draw_card_back(surface: pygame.Surface, x: int, y: int,
                   w: int = CARD_W, h: int = CARD_H) -> None:
    rect = pygame.Rect(x, y, w, h)
    shadow_rect = rect.move(3, 3)
    shadow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=7)
    surface.blit(shadow_surf, (shadow_rect.x, shadow_rect.y))
    pygame.draw.rect(surface, CARD_BACK_TOP, rect, border_radius=7)
    pygame.draw.rect(surface, (60, 80, 180), rect, 2, border_radius=7)
    inner = rect.inflate(-10, -10)
    pygame.draw.rect(surface, CARD_BACK_BOT, inner, border_radius=5)
    # Diamond pattern
    cx, cy = rect.centerx, rect.centery
    for di in range(-2, 3):
        for dj in range(-2, 3):
            px, py = cx + di * 12, cy + dj * 12
            if rect.inflate(-14, -14).collidepoint(px, py):
                pygame.draw.circle(surface, (50, 70, 160), (px, py), 3)


def centered_cards_x(n: int, center_x: int, card_w: int, gap: int = 6) -> int:
    total = n * card_w + (n - 1) * gap
    return center_x - total // 2


# ---------------------------------------------------------------------------
# Menu config
# ---------------------------------------------------------------------------

AGENT_OPTIONS = [
    "Random",
    "Heuristic (basic)",
    "Heuristic (aggressive)",
    "MCTS",
    "DNN",
]


@dataclass
class MenuConfig:
    # Which agents to include
    include_random:     bool = True
    include_heuristic_basic: bool = True
    include_heuristic_agg:   bool = True
    include_mcts:       bool = True
    include_dnn:        bool = False
    mcts_sims:          int  = 200
    dnn_model_path:     str  = "models/blackjack_mlp.pt"
    num_rounds:         int  = 20
    starting_bankroll:  float = 1000.0
    base_bet:           float = 10.0
    seed:               int   = 42

    def active_agents(self) -> list[str]:
        result = []
        if self.include_random:          result.append("Random")
        if self.include_heuristic_basic: result.append("Heuristic (basic)")
        if self.include_heuristic_agg:   result.append("Heuristic (aggressive)")
        if self.include_mcts:            result.append("MCTS")
        if self.include_dnn:             result.append("DNN")
        return result


def build_agent(name: str, cfg: MenuConfig) -> BaseAgent:
    n = name.lower()
    if n == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent()
    elif n == "heuristic (basic)":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="basic")
    elif n == "heuristic (aggressive)":
        from agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent(mode="aggressive")
    elif n == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(n_simulations=cfg.mcts_sims, n_determinizations=10)
    elif n == "dnn":
        p = cfg.dnn_model_path
        if not Path(p).exists():
            print(f"[WARNING] DNN model not found at '{p}'. Falling back to Heuristic.")
            from agents.heuristic_agent import HeuristicAgent
            return HeuristicAgent(mode="basic")
        from agents.dnn_agent import DNNAgent
        return DNNAgent(model_path=p)
    from agents.heuristic_agent import HeuristicAgent
    return HeuristicAgent(mode="basic")


# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------

@dataclass
class FrameSnapshot:
    """One display frame of the multi-agent game replay."""
    round_num: int
    dealer_upcard: Card
    dealer_hand_hidden: bool          # True = hole card still face down
    dealer_hand: list[Card]           # final hand (always available for lookup)
    # Per-agent state at this frame
    agent_hands: list[list[Card]]     # current hand for each agent
    agent_bets: list[float]
    agent_bankrolls: list[float]
    agent_names: list[str]
    # Decision highlight (None if no active decision this frame)
    active_agent_idx: int | None
    active_step: AgentRoundStep | None
    # Round-over state
    round_over: bool
    agent_results: list[AgentRoundResult] | None = None
    all_payouts: list[float] | None = None   # net payout per agent this round


def build_frames(result: MultiSessionResult) -> list[FrameSnapshot]:
    """
    Convert a MultiSessionResult into a list of FrameSnapshots for replay.
    Each agent decision becomes one frame; round-end is one frame per round.
    """
    frames: list[FrameSnapshot] = []
    n_agents = len(result.rounds[0].agent_results) if result.rounds else 0

    # Track running bankrolls
    bankrolls: list[float] = []
    bets: list[float] = []

    for rec in result.rounds:
        n = len(rec.agent_results)
        if not bankrolls:
            # Init from first round's starting bankroll (before bet)
            # Approximate: bankroll_after + net_payout = bankroll_before_final_payout
            for ar in rec.agent_results:
                bankrolls.append(ar.bankroll_after - ar.net_payout + ar.bets[0])
                bets.append(ar.bets[0])

        # Collect all decision steps across all agents for this round
        all_steps: list[tuple[int, AgentRoundStep]] = []
        for i, ar in enumerate(rec.agent_results):
            for step in ar.steps:
                all_steps.append((i, step))

        # Opening frame (before any decision): dealer upcard shown, hole hidden
        agent_hands = [ar.player_hands[0][:2] if ar.player_hands else []
                       for ar in rec.agent_results]
        agent_bets  = [ar.bets[0] if ar.bets else 0 for ar in rec.agent_results]
        agent_bks   = list(bankrolls)
        names = [ar.agent_name for ar in rec.agent_results]

        frames.append(FrameSnapshot(
            round_num=rec.round_num,
            dealer_upcard=rec.dealer_upcard,
            dealer_hand_hidden=True,
            dealer_hand=rec.dealer_hand,
            agent_hands=agent_hands,
            agent_bets=agent_bets,
            agent_bankrolls=list(agent_bks),
            agent_names=names,
            active_agent_idx=None,
            active_step=None,
            round_over=False,
        ))

        # One frame per decision step
        # Build a live copy of each agent's hand that grows as cards are dealt
        live_hands: list[list[Card]] = [list(h) for h in agent_hands]

        for agent_idx, step in all_steps:
            live_hands[agent_idx] = list(step.player_hand)
            frames.append(FrameSnapshot(
                round_num=rec.round_num,
                dealer_upcard=rec.dealer_upcard,
                dealer_hand_hidden=True,
                dealer_hand=rec.dealer_hand,
                agent_hands=[list(h) for h in live_hands],
                agent_bets=[ar.bets[0] for ar in rec.agent_results],
                agent_bankrolls=list(agent_bks),
                agent_names=names,
                active_agent_idx=agent_idx,
                active_step=step,
                round_over=False,
            ))

        # Update live_hands to final state
        for i, ar in enumerate(rec.agent_results):
            live_hands[i] = ar.player_hands[-1] if ar.player_hands else []

        # Round-over frame: reveal dealer, show payouts, update bankrolls
        for i, ar in enumerate(rec.agent_results):
            bankrolls[i] = ar.bankroll_after

        payouts = [ar.net_payout for ar in rec.agent_results]

        frames.append(FrameSnapshot(
            round_num=rec.round_num,
            dealer_upcard=rec.dealer_upcard,
            dealer_hand_hidden=False,
            dealer_hand=rec.dealer_hand,
            agent_hands=[list(ar.player_hands[-1]) if ar.player_hands else []
                         for ar in rec.agent_results],
            agent_bets=[ar.bets[0] for ar in rec.agent_results],
            agent_bankrolls=list(bankrolls),
            agent_names=names,
            active_agent_idx=None,
            active_step=None,
            round_over=True,
            agent_results=list(rec.agent_results),
            all_payouts=payouts,
        ))

    return frames


# ---------------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------------

class MainMenu:
    def __init__(self, screen: pygame.Surface, fonts: dict) -> None:
        self.screen = screen
        self.fonts = fonts
        self.cfg = MenuConfig()

        cx = W // 2
        col_left  = cx - 320
        col_right = cx + 60
        row_h = 56
        top = 180

        # Agent toggles (left column)
        self.cb_random   = CheckBox(pygame.Rect(col_left, top,           280, 40), "Random Agent",            fonts["sm"], True)
        self.cb_h_basic  = CheckBox(pygame.Rect(col_left, top + row_h,   280, 40), "Heuristic (Basic)",       fonts["sm"], True)
        self.cb_h_agg    = CheckBox(pygame.Rect(col_left, top + row_h*2, 280, 40), "Heuristic (Aggressive)",  fonts["sm"], True)
        self.cb_mcts     = CheckBox(pygame.Rect(col_left, top + row_h*3, 280, 40), "MCTS Agent",              fonts["sm"], True)
        self.cb_dnn      = CheckBox(pygame.Rect(col_left, top + row_h*4, 280, 40), "DNN Agent",               fonts["sm"], False)

        # Settings (right column)
        self.rounds_spin = NumberSpin(pygame.Rect(col_right, top,           220, 44), 20,   1,   200,  5, fonts["sm"])
        self.bet_spin    = NumberSpin(pygame.Rect(col_right, top + row_h,   220, 44), 10,   1,   500, 10, fonts["sm"])
        self.bk_spin     = NumberSpin(pygame.Rect(col_right, top + row_h*2, 220, 44), 1000, 100, 10000, 100, fonts["sm"])
        self.seed_spin   = NumberSpin(pygame.Rect(col_right, top + row_h*3, 220, 44), 42,   0,   9999,  1, fonts["sm"])
        self.mcts_spin   = NumberSpin(pygame.Rect(col_right, top + row_h*4, 220, 44), 200,  20,  1000, 20, fonts["sm"])
        self.dnn_input   = TextInput( pygame.Rect(col_right, top + row_h*5, 320, 40), fonts["xs"], "models/blackjack_mlp.pt")

        self.start_btn = Button(pygame.Rect(cx - 140, top + row_h*7, 280, 56), "▶  START TOURNAMENT", fonts["md"])
        self.quit_btn  = Button(pygame.Rect(cx - 60,  top + row_h*7 + 70, 120, 40), "Quit", fonts["sm"])

        self._checkboxes = [self.cb_random, self.cb_h_basic, self.cb_h_agg, self.cb_mcts, self.cb_dnn]
        self._spins = [self.rounds_spin, self.bet_spin, self.bk_spin, self.seed_spin, self.mcts_spin]

    def run(self) -> MenuConfig | None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return None
                for cb in self._checkboxes:
                    cb.handle_event(event)
                for sp in self._spins:
                    sp.handle_event(event)
                self.dnn_input.handle_event(event)
                if self.start_btn.handle_event(event):
                    cfg = self._build_cfg()
                    if cfg.active_agents():
                        return cfg
                if self.quit_btn.handle_event(event):
                    return None

            self._draw()
            clock.tick(60)

    def _build_cfg(self) -> MenuConfig:
        return MenuConfig(
            include_random=self.cb_random.checked,
            include_heuristic_basic=self.cb_h_basic.checked,
            include_heuristic_agg=self.cb_h_agg.checked,
            include_mcts=self.cb_mcts.checked,
            include_dnn=self.cb_dnn.checked,
            mcts_sims=self.mcts_spin.value,
            dnn_model_path=self.dnn_input.text,
            num_rounds=self.rounds_spin.value,
            starting_bankroll=float(self.bk_spin.value),
            base_bet=float(self.bet_spin.value),
            seed=self.seed_spin.value,
        )

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)
        fonts = self.fonts
        cx = W // 2

        # Title
        title = fonts["title"].render("♠  Blackjack AI Tournament  ♦", True, GOLD)
        self.screen.blit(title, title.get_rect(centerx=cx, y=36))
        sub = fonts["xs"].render(
            "ICOM/CIIC 5015 — Artificial Intelligence  |  All agents compete against the same dealer",
            True, TEXT_DIM
        )
        self.screen.blit(sub, sub.get_rect(centerx=cx, y=102))

        # Divider line
        pygame.draw.line(self.screen, PANEL_BORDER, (80, 130), (W - 80, 130), 1)

        col_left  = cx - 320
        col_right = cx + 60
        row_h = 56
        top = 180

        # Left column header
        lh = fonts["sm"].render("SELECT AGENTS", True, GOLD)
        self.screen.blit(lh, (col_left, top - 30))

        for cb in self._checkboxes:
            cb.draw(self.screen)

        # Agent color indicators
        colors = [AGENT_COLORS[0], AGENT_COLORS[1], AGENT_COLORS[2], AGENT_COLORS[3], AGENT_COLORS[4]]
        checkboxes_checked = [self.cb_random, self.cb_h_basic, self.cb_h_agg, self.cb_mcts, self.cb_dnn]
        for ci, (cb, col) in enumerate(zip(checkboxes_checked, colors)):
            dot_x = col_left - 16
            dot_y = cb.rect.centery
            if cb.checked:
                pygame.draw.circle(self.screen, col, (dot_x, dot_y), 6)
            else:
                pygame.draw.circle(self.screen, (80, 80, 80), (dot_x, dot_y), 6, 1)

        # Right column header
        rh = fonts["sm"].render("GAME SETTINGS", True, GOLD)
        self.screen.blit(rh, (col_right, top - 30))

        settings_labels = [
            "Number of Rounds",
            "Base Bet ($)",
            "Starting Bankroll ($)",
            "Random Seed",
            "MCTS Simulations",
            "DNN Model Path",
        ]
        for i, lbl in enumerate(settings_labels):
            t = fonts["xs"].render(lbl, True, TEXT_DIM)
            self.screen.blit(t, (col_right, top + row_h * i - 18))

        for sp in self._spins:
            sp.draw(self.screen)
        self.dnn_input.draw(self.screen)

        # DNN indicator
        exists = Path(self.dnn_input.text).exists()
        ind = fonts["xs"].render("✓ found" if exists else "✗ not found",
                                  True, GREEN_OK if exists else RED_ACCENT)
        self.screen.blit(ind, (col_right + 330, top + row_h * 5 + 12))

        self.start_btn.draw(self.screen)
        self.quit_btn.draw(self.screen)

        # Footer hint
        hint = fonts["xs"].render(
            "Tip: Enable multiple agents to see them compete head-to-head against the same dealer!",
            True, TEXT_DIM
        )
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=H - 30))

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Game Visualizer (multi-agent)
# ---------------------------------------------------------------------------

class GameVisualizer:
    """Displays the multi-agent replay frame by frame."""

    def __init__(self, screen: pygame.Surface, fonts: dict,
                 frames: list[FrameSnapshot], result: MultiSessionResult,
                 cfg: MenuConfig) -> None:
        self.screen = screen
        self.fonts  = fonts
        self.frames = frames
        self.result = result
        self.cfg    = cfg
        self.frame_idx  = 0
        self.auto_play  = True
        self.speed      = 1.2      # seconds per frame
        self._last_step = time.time()

        self.menu_btn = Button(pygame.Rect(10, 10, 110, 36), "← Menu", fonts["xs"])
        self.play_btn = Button(pygame.Rect(130, 10, 110, 36), "⏸ Pause", fonts["xs"])

        # Running tally for mini leaderboard in-game
        self._wins:   list[int] = [0] * len(result.leaderboard)
        self._losses: list[int] = [0] * len(result.leaderboard)
        self._pts:    list[int] = [0] * len(result.leaderboard)

        n = max(1, len(cfg.active_agents()))
        self.n_agents = n

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
                    if event.key == pygame.K_SPACE:
                        self.auto_play = not self.auto_play
                    if event.key == pygame.K_RIGHT:
                        self._advance()
                    if event.key == pygame.K_LEFT:
                        self.frame_idx = max(0, self.frame_idx - 1)
                    if event.key == pygame.K_RETURN:
                        self.frame_idx = len(self.frames) - 1
                        self.auto_play = False
                    if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.speed = max(0.1, self.speed / 1.5)
                    if event.key == pygame.K_MINUS:
                        self.speed = min(6.0, self.speed * 1.5)
                if self.menu_btn.handle_event(event):
                    return "menu"
                if self.play_btn.handle_event(event):
                    self.auto_play = not self.auto_play

            if self.auto_play and time.time() - self._last_step >= self.speed:
                if self.frame_idx < len(self.frames) - 1:
                    self._advance()
                else:
                    return "results"   # auto-advance to results when done

            self.play_btn.label = "⏸ Pause" if self.auto_play else "▶ Play"
            self._draw()
            clock.tick(60)

    def _advance(self) -> None:
        self.frame_idx = min(len(self.frames) - 1, self.frame_idx + 1)
        self._last_step = time.time()

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)
        if not self.frames:
            return

        frame = self.frames[self.frame_idx]
        fonts = self.fonts

        # ---- Header bar ----
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, 54))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, 54), (W, 54), 1)

        self.menu_btn.draw(self.screen)
        self.play_btn.draw(self.screen)

        hdr = fonts["md"].render(
            f"♠ Blackjack Tournament  |  Round {frame.round_num} / {self.cfg.num_rounds}",
            True, GOLD
        )
        self.screen.blit(hdr, hdr.get_rect(centerx=W // 2, centery=27))

        spd_txt = fonts["xs"].render(f"Speed {1/self.speed:.1f}x", True, TEXT_DIM)
        self.screen.blit(spd_txt, (W - 140, 8))
        ctr = fonts["xs"].render(f"Frame {self.frame_idx+1}/{len(self.frames)}", True, TEXT_DIM)
        self.screen.blit(ctr, (W - 140, 26))

        # ---- Dealer section (top center) ----
        self._draw_dealer(frame)

        # ---- Agent panels (bottom portion) ----
        self._draw_agents(frame)

        # ---- Active decision panel ----
        if frame.active_step is not None:
            self._draw_decision_panel(frame)

        # ---- Round-over overlay ----
        if frame.round_over and frame.all_payouts is not None:
            self._draw_round_results(frame)

        # ---- Controls hint ----
        hint = fonts["xs"].render(
            "SPACE: play/pause  |  ◄►: step  |  +/-: speed  |  ENTER: end  |  R: menu  |  Q: quit",
            True, (80, 130, 85)
        )
        self.screen.blit(hint, hint.get_rect(centerx=W // 2, y=H - 18))

        pygame.display.flip()

    def _draw_dealer(self, frame: FrameSnapshot) -> None:
        fonts = self.fonts
        dealer_cx = W // 2
        dy = 70

        lbl = fonts["sm"].render("D E A L E R", True, TEXT_DIM)
        self.screen.blit(lbl, lbl.get_rect(centerx=dealer_cx, y=dy))
        dy += 26

        if frame.dealer_hand_hidden:
            # Show upcard + face-down card
            sx = centered_cards_x(2, dealer_cx, CARD_W)
            draw_card(self.screen, frame.dealer_upcard, sx, dy,
                      fonts["card_lg"], fonts["card_sm"])
            draw_card_back(self.screen, sx + CARD_W + 6, dy)
            up_val = frame.dealer_upcard.value()
            val_txt = fonts["xs"].render(f"Showing: {up_val}", True, TEXT_DIM)
            self.screen.blit(val_txt, val_txt.get_rect(centerx=dealer_cx, y=dy + CARD_H + 6))
        else:
            # Reveal full hand
            n_cards = len(frame.dealer_hand)
            sx = centered_cards_x(n_cards, dealer_cx, CARD_W)
            for ci, c in enumerate(frame.dealer_hand):
                draw_card(self.screen, c, sx + ci * (CARD_W + 6), dy,
                          fonts["card_lg"], fonts["card_sm"])
            dval = hand_value(frame.dealer_hand)
            col = RED_ACCENT if dval > 21 else (GOLD if dval == 21 else WHITE)
            bust = " BUST!" if dval > 21 else (" BLACKJACK!" if is_blackjack(frame.dealer_hand) else "")
            val_txt = fonts["sm"].render(f"Value: {dval}{bust}", True, col)
            self.screen.blit(val_txt, val_txt.get_rect(centerx=dealer_cx, y=dy + CARD_H + 6))

    def _draw_agents(self, frame: FrameSnapshot) -> None:
        fonts = self.fonts
        n = len(frame.agent_names)
        if n == 0:
            return

        # Layout: n panels across the bottom area
        panel_area_top = 220
        panel_area_h   = H - 220 - 80   # leave room for controls hint
        panel_w = (W - 40) // n
        panel_gap = 4

        for i, name in enumerate(frame.agent_names):
            px = 20 + i * (panel_w + panel_gap)
            py = panel_area_top

            is_active = (frame.active_agent_idx == i)
            agent_col = AGENT_COLORS[i % len(AGENT_COLORS)]

            # Panel background
            panel_rect = pygame.Rect(px, py, panel_w, panel_area_h)
            bg_col = tuple(min(255, c + 12) for c in PANEL_BG) if is_active else PANEL_BG
            pygame.draw.rect(self.screen, bg_col, panel_rect, border_radius=10)
            border_col = agent_col if is_active else PANEL_BORDER
            border_w   = 2 if is_active else 1
            pygame.draw.rect(self.screen, border_col, panel_rect, border_w, border_radius=10)

            # Agent name strip
            strip = pygame.Rect(px, py, panel_w, 30)
            strip_col = tuple(c // 4 for c in agent_col)
            pygame.draw.rect(self.screen, strip_col, strip,
                             border_radius=10)
            # name
            name_surf = fonts["xs"].render(name, True, agent_col)
            self.screen.blit(name_surf, name_surf.get_rect(centerx=px + panel_w // 2, centery=py + 15))

            inner_y = py + 36

            # Bankroll & bet
            bk = frame.agent_bankrolls[i] if i < len(frame.agent_bankrolls) else 0
            bet = frame.agent_bets[i] if i < len(frame.agent_bets) else 0
            bk_surf = fonts["sm"].render(f"${bk:.0f}", True, GOLD)
            self.screen.blit(bk_surf, (px + 6, inner_y))
            bet_surf = fonts["xs"].render(f"Bet ${bet:.0f}", True, TEXT_DIM)
            self.screen.blit(bet_surf, (px + panel_w - bet_surf.get_width() - 6, inner_y + 2))
            inner_y += bk_surf.get_height() + 4

            # Player hand cards
            hand = frame.agent_hands[i] if i < len(frame.agent_hands) else []
            if hand:
                max_cards_visible = max(1, (panel_w - 12) // (CARD_SM_W + 4))
                n_show = min(len(hand), max_cards_visible)
                card_gap = min(6, (panel_w - 12 - CARD_SM_W) // max(n_show - 1, 1))
                cx_panel = px + panel_w // 2
                sx = centered_cards_x(n_show, cx_panel, CARD_SM_W, card_gap)
                for ci in range(n_show):
                    draw_card(self.screen, hand[ci],
                              sx + ci * (CARD_SM_W + card_gap), inner_y,
                              fonts["card_sm"], fonts["card_xs"],
                              w=CARD_SM_W, h=CARD_SM_H,
                              highlighted=is_active)
                inner_y += CARD_SM_H + 6

                # Hand value
                hv = hand_value(hand)
                hv_col = RED_ACCENT if hv > 21 else (GOLD if hv == 21 else WHITE)
                bj_note = " BJ!" if is_blackjack(hand) else (" BUST!" if hv > 21 else "")
                hv_surf = fonts["xs"].render(f"{hv}{bj_note}", True, hv_col)
                self.screen.blit(hv_surf, hv_surf.get_rect(centerx=px + panel_w // 2, y=inner_y))
                inner_y += hv_surf.get_height() + 4

            # Active agent: show legal actions + chosen action
            if is_active and frame.active_step is not None:
                step = frame.active_step
                # Legal actions
                for a in step.legal_actions:
                    col = GOLD if a == step.action_taken else TEXT_DIM
                    surf = fonts["xs"].render(f"[{a.upper()}]", True, col)
                    self.screen.blit(surf, surf.get_rect(centerx=px + panel_w // 2, y=inner_y))
                    inner_y += surf.get_height() + 1

                # Chosen action badge
                act_surf = fonts["sm"].render(step.action_taken.upper(), True, BLACK)
                badge_rect = act_surf.get_rect(centerx=px + panel_w // 2, centery=inner_y + 14)
                badge_bg = badge_rect.inflate(14, 6)
                pygame.draw.rect(self.screen, agent_col, badge_bg, border_radius=6)
                self.screen.blit(act_surf, badge_rect)
                inner_y += badge_bg.height + 6

                # Reason (word-wrapped to panel width)
                reason_lines = _wrap_text(step.reason, fonts["card_xs"], panel_w - 12)
                for line in reason_lines[:3]:
                    ls = fonts["card_xs"].render(line, True, TEXT_DIM)
                    self.screen.blit(ls, (px + 6, inner_y))
                    inner_y += ls.get_height() + 1

    def _draw_decision_panel(self, frame: FrameSnapshot) -> None:
        """Bottom banner showing the active agent's decision details."""
        if frame.active_step is None:
            return
        step = frame.active_step
        fonts = self.fonts
        agent_col = AGENT_COLORS[step.agent_index % len(AGENT_COLORS)]

        bar_h = 56
        bar_rect = pygame.Rect(0, H - 80, W, bar_h)
        pygame.draw.rect(self.screen, FELT_DARK, bar_rect)
        pygame.draw.line(self.screen, agent_col, (0, H - 80), (W, H - 80), 2)

        # Agent name
        n_surf = fonts["sm"].render(f"{step.agent_name}", True, agent_col)
        self.screen.blit(n_surf, (16, H - 72))

        # Action
        a_surf = fonts["lg"].render(step.action_taken.upper(), True, GOLD)
        self.screen.blit(a_surf, (16, H - 52))

        # Reason
        r_surf = fonts["xs"].render(f"Reason: {step.reason}", True, TEXT_MED)
        self.screen.blit(r_surf, (200, H - 58))

        # Hand value
        hv = step.hand_value
        hv_col = RED_ACCENT if hv > 21 else WHITE
        hv_surf = fonts["xs"].render(f"Hand: {hv}", True, hv_col)
        self.screen.blit(hv_surf, (200, H - 40))

        # Dealer upcard
        du_surf = fonts["xs"].render(f"Dealer shows: {step.dealer_upcard}", True, TEXT_DIM)
        self.screen.blit(du_surf, (350, H - 40))

    def _draw_round_results(self, frame: FrameSnapshot) -> None:
        """Show a subtle result overlay across all agent panels when round ends."""
        if frame.all_payouts is None:
            return
        fonts = self.fonts
        n = len(frame.agent_names)
        panel_w = (W - 40) // n
        panel_gap = 4
        panel_area_top = 220

        for i, payout in enumerate(frame.all_payouts):
            px = 20 + i * (panel_w + panel_gap)
            if payout > 0:
                txt, col = f"+${payout:.0f} WIN", GREEN_OK
            elif payout < 0:
                txt, col = f"-${abs(payout):.0f} LOSS", RED_ACCENT
            else:
                txt, col = "PUSH", AMBER

            surf = fonts["md"].render(txt, True, col)
            self.screen.blit(surf, surf.get_rect(centerx=px + panel_w // 2,
                                                   centery=panel_area_top + 170))


# ---------------------------------------------------------------------------
# Leaderboard / Results Screen
# ---------------------------------------------------------------------------

class LeaderboardScreen:
    def __init__(self, screen: pygame.Surface, fonts: dict,
                 result: MultiSessionResult, cfg: MenuConfig) -> None:
        self.screen = screen
        self.fonts  = fonts
        self.result = result
        self.cfg    = cfg
        cx = W // 2
        self.play_btn  = Button(pygame.Rect(cx - 280, H - 100, 180, 50), "▶ Play Again", fonts["md"])
        self.menu_btn  = Button(pygame.Rect(cx - 80,  H - 100, 160, 50), "Main Menu",    fonts["md"])
        self.quit_btn  = Button(pygame.Rect(cx + 100, H - 100, 120, 50), "Quit",         fonts["sm"])
        self._slide = 0.0
        # Pre-build history data per agent
        self._histories = self._build_histories()

    def _build_histories(self) -> list[list[float]]:
        lb = self.result.leaderboard
        histories: list[list[float]] = [[] for _ in lb]
        start = self.cfg.starting_bankroll
        for rec in self.result.rounds:
            for entry in lb:
                i = entry.agent_index
                if i < len(rec.agent_results):
                    ar = rec.agent_results[i]
                    histories[entry.agent_index].append(ar.bankroll_after)
        return histories

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
                if self.play_btn.handle_event(event):
                    return "play_again"
                if self.menu_btn.handle_event(event):
                    return "menu"
                if self.quit_btn.handle_event(event):
                    return "quit"

            self._slide = min(1.0, self._slide + 0.04)
            self._draw()
            clock.tick(60)

    def _draw(self) -> None:
        oy = int((1.0 - self._slide) * H)
        self.screen.fill(BG_COLOR)
        fonts = self.fonts
        cx = W // 2

        # Title
        title = fonts["title"].render("🏆  Tournament Results  🏆", True, GOLD)
        self.screen.blit(title, title.get_rect(centerx=cx, y=28 + oy))

        sub = fonts["xs"].render(
            f"{self.result.rounds_played} rounds  |  "
            f"Base Bet ${self.cfg.base_bet:.0f}  |  Starting ${self.cfg.starting_bankroll:.0f}",
            True, TEXT_DIM
        )
        self.screen.blit(sub, sub.get_rect(centerx=cx, y=88 + oy))

        pygame.draw.line(self.screen, PANEL_BORDER, (60, 110 + oy), (W - 60, 110 + oy), 1)

        lb = self.result.leaderboard

        # ---- Leaderboard table ----
        table_top = 120 + oy
        col_x = [80, 260, 390, 470, 550, 640, 730, 840]
        headers = ["Rank", "Agent", "Points", "Wins", "Losses", "Ties", "Win%", "Net Profit"]
        for ci, hdr in enumerate(headers):
            hs = fonts["xs"].render(hdr, True, GOLD)
            self.screen.blit(hs, (col_x[ci], table_top))

        pygame.draw.line(self.screen, PANEL_BORDER, (60, table_top + 20), (W - 60, table_top + 20), 1)

        medal_colors = [GOLD, SILVER, BRONZE]

        for ri, entry in enumerate(lb):
            ry = table_top + 26 + ri * 38

            # Row background
            row_col = PANEL_BG2 if ri % 2 == 0 else PANEL_BG
            pygame.draw.rect(self.screen, row_col, pygame.Rect(62, ry - 4, W - 124, 34), border_radius=5)

            # Rank medal
            rank_num = ri + 1
            medal_col = medal_colors[ri] if ri < 3 else TEXT_DIM
            rank_s = fonts["md"].render(f"#{rank_num}", True, medal_col)
            self.screen.blit(rank_s, (col_x[0], ry))

            # Agent name with colour dot
            agent_col = AGENT_COLORS[entry.agent_index % len(AGENT_COLORS)]
            pygame.draw.circle(self.screen, agent_col, (col_x[1] - 12, ry + 10), 5)
            name_s = fonts["sm"].render(entry.name, True, agent_col)
            self.screen.blit(name_s, (col_x[1], ry))

            # Points
            pts_s = fonts["md"].render(str(entry.points), True, WHITE)
            self.screen.blit(pts_s, (col_x[2], ry))

            # W/L/T
            self.screen.blit(fonts["sm"].render(str(entry.wins),   True, GREEN_OK),   (col_x[3], ry))
            self.screen.blit(fonts["sm"].render(str(entry.losses), True, RED_ACCENT), (col_x[4], ry))
            self.screen.blit(fonts["sm"].render(str(entry.ties),   True, AMBER),      (col_x[5], ry))

            # Win rate
            wr_col = GREEN_OK if entry.win_rate >= 0.5 else RED_ACCENT
            self.screen.blit(fonts["sm"].render(f"{entry.win_rate:.1%}", True, wr_col), (col_x[6], ry))

            # Net profit
            profit = entry.net_profit
            pc = GREEN_OK if profit >= 0 else RED_ACCENT
            self.screen.blit(fonts["sm"].render(f"${profit:+.0f}", True, pc), (col_x[7], ry))

        # ---- Bankroll chart (right side) ----
        chart_right = W - 60
        chart_left  = 960
        chart_top   = table_top + 26 + oy
        chart_bot   = H - 130
        chart_rect  = pygame.Rect(chart_left, chart_top, chart_right - chart_left, chart_bot - chart_top)
        pygame.draw.rect(self.screen, PANEL_BG, chart_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, chart_rect, 1, border_radius=8)

        ch = fonts["xs"].render("Bankroll History", True, GOLD)
        self.screen.blit(ch, (chart_left + 8, chart_top + 6))

        inner = chart_rect.inflate(-20, -30)
        inner.y += 24

        all_bks: list[float] = []
        for h in self._histories:
            all_bks.extend(h)
        all_bks.append(self.cfg.starting_bankroll)
        if all_bks:
            min_b = min(all_bks)
            max_b = max(all_bks)
            rng   = max(max_b - min_b, 1.0)

            def tp(idx: int, total: int, b: float) -> tuple[int, int]:
                x = inner.x + int(idx / max(total - 1, 1) * inner.width)
                y = inner.bottom - int((b - min_b) / rng * inner.height)
                return x, max(inner.y, min(inner.bottom, y))

            for entry in lb:
                hist = self._histories[entry.agent_index]
                if len(hist) < 2:
                    continue
                col = AGENT_COLORS[entry.agent_index % len(AGENT_COLORS)]
                pts = [tp(i, len(hist), b) for i, b in enumerate(hist)]
                pygame.draw.lines(self.screen, col, False, pts, 2)

            # Start bankroll line
            sy = inner.bottom - int((self.cfg.starting_bankroll - min_b) / rng * inner.height)
            sy = max(inner.y, min(inner.bottom, sy))
            pygame.draw.line(self.screen, (120, 120, 120), (inner.x, sy), (inner.right, sy), 1)

            # Legend
            legend_y = chart_top + 8
            for entry in lb:
                col = AGENT_COLORS[entry.agent_index % len(AGENT_COLORS)]
                pygame.draw.line(self.screen, col, (chart_right - 130, legend_y + 5),
                                 (chart_right - 105, legend_y + 5), 2)
                ls = fonts["card_xs"].render(entry.name, True, col)
                self.screen.blit(ls, (chart_right - 100, legend_y))
                legend_y += ls.get_height() + 3

        # ---- Point system explanation ----
        pts_panel = pygame.Rect(62, chart_top, chart_left - 80, 50)
        pts_lbl = fonts["xs"].render(
            "Points: WIN=3  |  TIE=1  |  LOSS=0  |  BLACKJACK bonus=+2",
            True, TEXT_DIM
        )
        self.screen.blit(pts_lbl, (62, chart_bot - 10 + oy))

        self.play_btn.draw(self.screen)
        self.menu_btn.draw(self.screen)
        self.quit_btn.draw(self.screen)

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------

def show_loading(screen: pygame.Surface, fonts: dict, agents: list[str]) -> None:
    screen.fill(BG_COLOR)
    msg = fonts["lg"].render("Simulating Tournament...", True, GOLD)
    screen.blit(msg, msg.get_rect(center=(W // 2, H // 2 - 40)))
    agents_str = "  vs  ".join(agents) if agents else "agents"
    sub = fonts["sm"].render(agents_str, True, TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 10)))
    hint = fonts["xs"].render("Please wait...", True, (80, 130, 85))
    screen.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 50)))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Text wrap helper
# ---------------------------------------------------------------------------

def _wrap_text(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        test = current + (" " if current else "") + w
        if font.size(test)[0] <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Blackjack AI Tournament — ICOM/CIIC 5015")

    try:
        title_font   = pygame.font.SysFont("segoeui", 48, bold=True)
        lg_font      = pygame.font.SysFont("segoeui", 32, bold=True)
        md_font      = pygame.font.SysFont("segoeui", 22, bold=True)
        sm_font      = pygame.font.SysFont("segoeui", 17)
        xs_font      = pygame.font.SysFont("segoeui", 13)
        card_lg_font = pygame.font.SysFont("segoeui", 20, bold=True)
        card_sm_font = pygame.font.SysFont("segoeui", 13)
        card_xs_font = pygame.font.SysFont("segoeui", 11)
    except Exception:
        title_font   = pygame.font.Font(None, 58)
        lg_font      = pygame.font.Font(None, 40)
        md_font      = pygame.font.Font(None, 28)
        sm_font      = pygame.font.Font(None, 22)
        xs_font      = pygame.font.Font(None, 16)
        card_lg_font = pygame.font.Font(None, 24)
        card_sm_font = pygame.font.Font(None, 16)
        card_xs_font = pygame.font.Font(None, 14)

    fonts = {
        "title":   title_font,
        "lg":      lg_font,
        "md":      md_font,
        "sm":      sm_font,
        "xs":      xs_font,
        "card_lg": card_lg_font,
        "card_sm": card_sm_font,
        "card_xs": card_xs_font,
    }

    while True:
        # ---- Main menu ----
        menu = MainMenu(screen, fonts)
        cfg = menu.run()
        if cfg is None:
            break

        active_names = cfg.active_agents()
        if not active_names:
            continue

        # ---- Loading ----
        show_loading(screen, fonts, active_names)
        pygame.event.pump()

        # ---- Build agents & run game ----
        agents = [build_agent(n, cfg) for n in active_names]
        game = MultiAgentGame(
            agents=agents,
            num_rounds=cfg.num_rounds,
            starting_bankroll=cfg.starting_bankroll,
            base_bet=cfg.base_bet,
            seed=cfg.seed,
        )
        result = game.run()
        frames = build_frames(result)

        if not frames:
            continue

        # ---- Game visualizer ----
        viz = GameVisualizer(screen, fonts, frames, result, cfg)
        outcome = viz.run()
        if outcome == "quit":
            break
        if outcome == "menu":
            continue

        # ---- Leaderboard ----
        lb_screen = LeaderboardScreen(screen, fonts, result, cfg)
        lb_outcome = lb_screen.run()
        if lb_outcome == "quit":
            break
        elif lb_outcome == "play_again":
            # Re-run same config
            show_loading(screen, fonts, active_names)
            pygame.event.pump()
            agents = [build_agent(n, cfg) for n in active_names]
            game = MultiAgentGame(
                agents=agents,
                num_rounds=cfg.num_rounds,
                starting_bankroll=cfg.starting_bankroll,
                base_bet=cfg.base_bet,
                seed=cfg.seed + 1,
            )
            result = game.run()
            frames = build_frames(result)
            viz = GameVisualizer(screen, fonts, frames, result, cfg)
            outcome2 = viz.run()
            if outcome2 == "quit":
                break
        # else "menu" → loop back

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
