import { useSession } from '../context/SessionContext';
import { BankrollChart } from '../components/leaderboard/BankrollChart';
import { AGENT_COLORS } from '../types/session';
import type { LeaderboardEntry } from '../types/session';

const MEDALS = ['🥇', '🥈', '🥉'];

function LeaderboardRow({ entry, rank }: { entry: LeaderboardEntry; rank: number }) {
  const color  = AGENT_COLORS[entry.name] ?? '#C9A962';
  const profit = entry.net_profit;
  const isFirst = rank === 0;

  return (
    <div
      className="flex items-center px-5 lb-slide"
      style={{
        height:          '56px',
        borderBottom:    '1px solid #2A2A2A',
        background:      isFirst ? 'rgba(201,169,98,0.04)' : 'transparent',
        animationDelay:  `${rank * 60}ms`,
      }}
    >
      {/* Rank */}
      <div className="w-8 flex-shrink-0">
        <span
          className="text-sm font-medium"
          style={{ color: isFirst ? '#C9A962' : '#848484', fontFamily: 'JetBrains Mono, monospace' }}
        >{rank + 1}</span>
      </div>

      {/* Color dot + name */}
      <div className="flex items-center gap-3 w-52 flex-shrink-0">
        <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
        <span className="text-sm font-medium text-white truncate">{entry.name}</span>
        {rank < 3 && <span className="text-base">{MEDALS[rank]}</span>}
      </div>

      {/* Wins */}
      <div className="w-12 flex-shrink-0">
        <span className="text-sm font-medium text-white">{entry.wins}</span>
      </div>

      {/* Losses */}
      <div className="w-12 flex-shrink-0">
        <span className="text-sm text-[#848484]">{entry.losses}</span>
      </div>

      {/* Points */}
      <div className="w-20 flex-shrink-0">
        <span
          className="text-sm font-medium"
          style={{ color: isFirst ? '#C9A962' : '#FFFFFF' }}
        >{entry.points}</span>
      </div>

      {/* Win Rate */}
      <div className="w-20 flex-shrink-0">
        <span className="text-sm text-white">{(entry.win_rate * 100).toFixed(1)}%</span>
      </div>

      {/* Final bankroll */}
      <div className="w-28 flex-shrink-0">
        <span
          className="text-sm font-medium"
          style={{ color: isFirst ? '#C9A962' : '#FFFFFF', fontFamily: 'JetBrains Mono, monospace' }}
        >${entry.final_bankroll.toFixed(0)}</span>
      </div>

      {/* Net profit */}
      <div className="flex-1">
        <span
          className="text-sm font-medium"
          style={{ color: profit >= 0 ? '#C9A962' : '#f87171', fontFamily: 'JetBrains Mono, monospace' }}
        >{profit >= 0 ? '+' : ''}${profit.toFixed(0)}</span>
      </div>
    </div>
  );
}

export function LeaderboardView() {
  const { state, dispatch } = useSession();
  const session = state.session!;
  const lb = session.leaderboard;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#0A0A0A' }}>

      {/* Header */}
      <header
        className="flex items-center justify-between px-14 h-[72px] flex-shrink-0"
        style={{ borderBottom: '1px solid #2A2A2A' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 flex items-center justify-center"
            style={{ border: '1px solid #C9A962', color: '#C9A962', fontFamily: 'Cormorant Garamond, serif', fontSize: '18px', fontWeight: 600 }}
          >♠</div>
          <span className="text-sm font-medium tracking-[3px] text-white">BLACKJACK AI</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => dispatch({ type: 'GO_TO_GAME' })}
            className="h-9 px-4 text-xs text-[#848484] hover:text-white transition cursor-pointer"
            style={{ border: '1px solid #2A2A2A' }}
          >◄ Replay</button>
          <button
            onClick={() => dispatch({ type: 'GO_TO_MENU' })}
            className="h-9 px-4 text-xs font-medium cursor-pointer"
            style={{ background: '#C9A962', color: '#0A0A0A' }}
          >New Game</button>
        </div>
      </header>

      {/* Hero */}
      <div
        className="flex flex-col items-center py-8 gap-2"
        style={{ borderBottom: '1px solid #2A2A2A' }}
      >
        <span className="text-[11px] font-medium tracking-[3px] text-[#848484]">TOURNAMENT COMPLETE</span>
        <h1 className="text-5xl font-medium text-white" style={{ fontFamily: 'Cormorant Garamond, serif' }}>
          Final Standings
        </h1>
        <p className="text-sm text-[#4A4A4A]">
          {session.rounds_played} rounds played · ${session.starting_bankroll} starting bankroll ·
          WIN=3pts · TIE=1pt · BLACKJACK+2pts
        </p>
      </div>

      {/* Body: two-column */}
      <div className="flex flex-1 overflow-auto">

        {/* Left — Podium */}
        <aside
          className="flex flex-col gap-5 px-10 py-8 flex-shrink-0"
          style={{ width: '360px', borderRight: '1px solid #2A2A2A' }}
        >
          <span className="text-[11px] font-medium tracking-[3px] text-[#848484]">PODIUM</span>
          {lb.slice(0, 3).map((entry, rank) => {
            const profit = entry.net_profit;
            return (
              <div
                key={entry.name}
                className="flex items-center gap-4 px-5 lb-slide"
                style={{
                  height:          rank === 0 ? '88px' : '76px',
                  border:          rank === 0 ? '1px solid #C9A962' : '1px solid #2A2A2A',
                  background:      rank === 0 ? 'rgba(201,169,98,0.06)' : 'transparent',
                  animationDelay:  `${rank * 100}ms`,
                }}
              >
                <span className="text-2xl flex-shrink-0">{MEDALS[rank]}</span>
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  <span
                    className="font-medium truncate"
                    style={{ fontFamily: 'Cormorant Garamond, serif', fontSize: rank === 0 ? '18px' : '16px', color: '#FFFFFF' }}
                  >{entry.name}</span>
                  <span className="text-xs text-[#848484]">
                    {entry.points} pts · {entry.wins}W {entry.losses}L
                  </span>
                </div>
                <div className="flex flex-col items-end flex-shrink-0">
                  <span className="text-[10px] tracking-[2px] text-[#848484]">FINAL</span>
                  <span
                    className="font-medium"
                    style={{ fontFamily: 'Cormorant Garamond, serif', fontSize: rank === 0 ? '20px' : '17px', color: rank === 0 ? '#C9A962' : '#848484' }}
                  >${entry.final_bankroll.toFixed(0)}</span>
                  <span
                    className="text-xs"
                    style={{ color: profit >= 0 ? '#C9A962' : '#f87171', fontFamily: 'JetBrains Mono, monospace' }}
                  >{profit >= 0 ? '+' : ''}${profit.toFixed(0)}</span>
                </div>
              </div>
            );
          })}

          {/* Bankroll chart */}
          <div className="mt-4">
            <span className="text-[11px] font-medium tracking-[3px] text-[#848484] block mb-3">BANKROLL HISTORY</span>
            <BankrollChart session={session} />
          </div>
        </aside>

        {/* Right — Full table */}
        <main className="flex flex-col flex-1 px-8 py-8">
          {/* Table header */}
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-2xl font-medium text-white" style={{ fontFamily: 'Cormorant Garamond, serif' }}>
              Full Results
            </h2>
          </div>

          {/* Column headers */}
          <div
            className="flex items-center px-5 text-xs text-[#848484] flex-shrink-0"
            style={{ height: '44px', borderBottom: '1px solid #2A2A2A', background: '#0D0D0D' }}
          >
            <div className="w-8 flex-shrink-0">#</div>
            <div className="w-52 flex-shrink-0">AGENT</div>
            <div className="w-12 flex-shrink-0">W</div>
            <div className="w-12 flex-shrink-0">L</div>
            <div className="w-20 flex-shrink-0">POINTS</div>
            <div className="w-20 flex-shrink-0">WIN %</div>
            <div className="w-28 flex-shrink-0">FINAL $</div>
            <div className="flex-1">NET</div>
          </div>

          {/* Rows */}
          <div style={{ border: '1px solid #2A2A2A', borderTop: 'none' }}>
            {lb.map((entry, rank) => (
              <LeaderboardRow key={entry.name} entry={entry} rank={rank} />
            ))}
          </div>

          {/* Round log */}
          <div className="mt-8">
            <span className="text-[11px] font-medium tracking-[3px] text-[#848484] block mb-3">
              LAST {Math.min(15, session.rounds.length)} ROUNDS
            </span>
            <RoundLog session={session} />
          </div>
        </main>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Round log
// ---------------------------------------------------------------------------

function RoundLog({ session }: { session: import('../types/session').SessionResult }) {
  const rounds = session.rounds.slice(-15);
  return (
    <div style={{ border: '1px solid #2A2A2A', overflow: 'auto' }}>
      <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#0D0D0D', borderBottom: '1px solid #2A2A2A' }}>
            <th className="text-left px-4 py-3 text-[#848484] font-normal">Rnd</th>
            <th className="text-left px-4 py-3 text-[#848484] font-normal">Dealer</th>
            {session.rounds[0]?.agent_results.map(ar => (
              <th key={ar.agent_name} className="text-left px-4 py-3 font-medium" style={{ color: AGENT_COLORS[ar.agent_name] ?? '#848484' }}>
                {ar.agent_name.split('(')[0]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rounds.map(round => (
            <tr key={round.round_num} style={{ borderBottom: '1px solid #1A1A1A' }}>
              <td className="px-4 py-2.5 text-white" style={{ fontFamily: 'JetBrains Mono, monospace' }}>{round.round_num}</td>
              <td className="px-4 py-2.5 text-[#848484]" style={{ fontFamily: 'JetBrains Mono, monospace' }}>{round.dealer_hand.join(' ')}</td>
              {round.agent_results.map(ar => {
                const p = ar.net_payout;
                return (
                  <td key={ar.agent_name} className="px-4 py-2.5 font-medium" style={{
                    color: p > 0 ? '#C9A962' : p < 0 ? '#f87171' : '#848484',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    {p > 0 ? `+$${p}` : p < 0 ? `-$${Math.abs(p)}` : 'PUSH'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
