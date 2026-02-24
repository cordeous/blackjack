import type { AgentRoundStep } from '../../types/session';
import { AGENT_COLORS, ACTION_COLORS } from '../../types/session';
import { HandDisplay } from './HandDisplay';

interface Props {
  name:             string;
  currentHand:      string[];
  bankroll:         number;
  bet:              number;
  isActive:         boolean;
  activeStep:       AgentRoundStep | null;
  payout:           number | null;
  startingBankroll: number;
}

function handValue(cards: string[]): number {
  let total = 0, aces = 0;
  for (const c of cards) {
    const rank = c.slice(0, -1);
    if (rank === 'A') { total += 11; aces++; }
    else if (['J','Q','K'].includes(rank)) total += 10;
    else total += parseInt(rank, 10) || 0;
  }
  while (total > 21 && aces > 0) { total -= 10; aces--; }
  return total;
}

function isBlackjack(cards: string[]): boolean {
  return cards.length === 2 && handValue(cards) === 21;
}

const ACTION_BG: Record<string, string> = {
  hit:    'rgba(22,163,74,0.15)',
  stand:  'rgba(220,38,38,0.15)',
  double: 'rgba(217,119,6,0.15)',
  split:  'rgba(124,58,237,0.15)',
};
const ACTION_FG: Record<string, string> = {
  hit:    '#4ade80',
  stand:  '#f87171',
  double: '#fbbf24',
  split:  '#c4b5fd',
};

export function AgentPanel({ name, currentHand, bankroll, bet, isActive, activeStep, payout, startingBankroll }: Props) {
  const color  = AGENT_COLORS[name] ?? '#C9A962';
  const val    = currentHand.length > 0 ? handValue(currentHand) : null;
  const bust   = val !== null && val > 21;
  const bj     = val !== null && isBlackjack(currentHand);
  const profit = bankroll - startingBankroll;

  return (
    <div
      className="flex flex-col transition-all duration-300 overflow-hidden"
      style={{
        border:     `1px solid ${isActive ? color : '#2A2A2A'}`,
        background: isActive ? `${color}08` : '#0A0A0A',
        boxShadow:  isActive ? `inset 0 0 0 1px ${color}44` : 'none',
      }}
    >
      {/* Agent name strip */}
      <div
        className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: `1px solid ${isActive ? color + '40' : '#1A1A1A'}` }}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
          <span className="text-xs font-medium truncate" style={{ color: isActive ? color : '#FFFFFF' }}>{name}</span>
          {isActive && (
            <span
              className="text-[9px] font-medium px-2 py-0.5"
              style={{ border: `1px solid ${color}66`, color }}
            >ACTIVE</span>
          )}
        </div>
        <div className="flex flex-col items-end">
          <span
            className="text-sm font-medium"
            style={{ fontFamily: 'JetBrains Mono, monospace', color: '#C9A962' }}
          >${bankroll.toFixed(0)}</span>
          <span
            className="text-[10px]"
            style={{ color: profit >= 0 ? '#C9A962' : '#f87171', fontFamily: 'JetBrains Mono, monospace' }}
          >{profit >= 0 ? '+' : ''}{profit.toFixed(0)}</span>
        </div>
      </div>

      {/* Cards */}
      <div className="flex-1 flex flex-col items-center justify-center py-4 px-3 gap-3 min-h-[120px]">
        {currentHand.length > 0 ? (
          <HandDisplay
            cards={currentHand}
            small
            highlight={isActive}
            showValue
            value={val ?? 0}
            isBust={bust}
            isBlackjack={bj}
          />
        ) : (
          <span className="text-xs text-[#4A4A4A]">Waiting…</span>
        )}
        <div className="text-[10px] text-[#848484]">
          Bet <span className="text-white" style={{ fontFamily: 'JetBrains Mono, monospace' }}>${bet}</span>
        </div>
      </div>

      {/* Decision area */}
      <div className="px-4 pb-4 min-h-[72px] flex flex-col justify-end">
        {payout !== null ? (
          <div
            className="flex items-center justify-center h-9 text-sm font-bold tracking-widest slide-up"
            style={{
              background: payout > 0 ? 'rgba(22,163,74,0.15)' : payout < 0 ? 'rgba(220,38,38,0.15)' : 'rgba(217,119,6,0.15)',
              border:     `1px solid ${payout > 0 ? '#4ade80' : payout < 0 ? '#f87171' : '#fbbf24'}`,
              color:      payout > 0 ? '#4ade80'  : payout < 0 ? '#f87171'  : '#fbbf24',
            }}
          >
            {payout > 0 ? `+$${payout.toFixed(0)} WIN` : payout < 0 ? `-$${Math.abs(payout).toFixed(0)} LOSS` : 'PUSH'}
          </div>
        ) : activeStep ? (
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap gap-1 justify-center">
              {activeStep.legal_actions.map(a => (
                <span
                  key={a}
                  className="text-[9px] px-2 py-1 font-bold uppercase tracking-widest"
                  style={{
                    background: a === activeStep.action_taken ? (ACTION_BG[a] ?? 'rgba(255,255,255,0.1)') : 'transparent',
                    color:      a === activeStep.action_taken ? (ACTION_FG[a] ?? '#FFFFFF') : '#4A4A4A',
                    border:     `1px solid ${a === activeStep.action_taken ? (ACTION_FG[a] ?? '#FFFFFF') + '66' : '#2A2A2A'}`,
                  }}
                >{a}</span>
              ))}
            </div>
            <p className="text-[10px] text-[#848484] text-center leading-snug line-clamp-2">
              {activeStep.reason}
            </p>
          </div>
        ) : (
          <div className="flex justify-center">
            <span className="text-[10px] text-[#2A2A2A]">—</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Re-export for GameView usage
export { ACTION_COLORS };
