"""Statistical metrics for Blackjack agent evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AgentStats:
    """Accumulate per-round outcomes for one agent configuration."""
    agent_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_payout: list[float] = field(default_factory=list)
    decision_times_ms: list[float] = field(default_factory=list)
    bankrolls: list[float] = field(default_factory=list)

    @property
    def total_rounds(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_rounds, 1)

    @property
    def avg_payout(self) -> float:
        return sum(self.total_payout) / max(len(self.total_payout), 1)

    @property
    def std_payout(self) -> float:
        n = len(self.total_payout)
        if n < 2:
            return 0.0
        mean = self.avg_payout
        var = sum((x - mean) ** 2 for x in self.total_payout) / (n - 1)
        return math.sqrt(var)

    @property
    def avg_decision_time_ms(self) -> float:
        return sum(self.decision_times_ms) / max(len(self.decision_times_ms), 1)

    def confidence_interval_win_rate(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for win rate at given z (default 95%)."""
        n = self.total_rounds
        if n == 0:
            return (0.0, 0.0)
        p = self.win_rate
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = (z / (1 + z**2 / n)) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (max(0.0, center - half), min(1.0, center + half))

    def confidence_interval_payout(self, z: float = 1.96) -> tuple[float, float]:
        """Student t-interval for average payout."""
        n = len(self.total_payout)
        if n < 2:
            return (self.avg_payout, self.avg_payout)
        sem = self.std_payout / math.sqrt(n)
        return (self.avg_payout - z * sem, self.avg_payout + z * sem)

    def final_bankroll(self) -> float:
        return self.bankrolls[-1] if self.bankrolls else 0.0

    def net_profit(self) -> float:
        return sum(self.total_payout)


@dataclass
class EvaluationReport:
    """Aggregated results for a comparison run."""
    stats: list[AgentStats]
    num_sessions: int
    rounds_per_session: int

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print(f"{'BLACKJACK EVALUATION REPORT':^65}")
        print(f"Sessions: {self.num_sessions}  |  Rounds/session: {self.rounds_per_session}")
        print("=" * 65)

        for s in self.stats:
            lo_wr, hi_wr = s.confidence_interval_win_rate()
            lo_p, hi_p = s.confidence_interval_payout()
            print(f"\n  Agent: {s.agent_name}")
            print(f"    Win rate:       {s.win_rate:.3f}  (95% CI [{lo_wr:.3f}, {hi_wr:.3f}])")
            print(f"    Avg payout/rnd: {s.avg_payout:+.2f}  (95% CI [{lo_p:+.2f}, {hi_p:+.2f}])")
            print(f"    Net profit:     {s.net_profit():+.1f}")
            print(f"    Std payout:     {s.std_payout:.2f}")
            print(f"    Avg decision:   {s.avg_decision_time_ms:.1f} ms")
            print(f"    W/L/T:          {s.wins}/{s.losses}/{s.ties}")

        print("\n" + "=" * 65)
