import { SUIT_COLORS } from '../../types/session';

interface Props {
  card:        string;     // e.g. "A♠", "10♥"
  faceDown?:   boolean;
  small?:      boolean;
  highlight?:  boolean;
  animDelay?:  number;     // ms delay for flip animation
}

function parseCard(card: string): { rank: string; symbol: string } {
  const last = card.slice(-1);
  const suitSymbols = ['♥', '♦', '♣', '♠'];
  if (suitSymbols.includes(last)) {
    return { rank: card.slice(0, -1), symbol: last };
  }
  return { rank: card, symbol: '' };
}

export function CardDisplay({ card, faceDown = false, small = false, highlight = false, animDelay = 0 }: Props) {
  const w = small ? 'w-9 h-13' : 'w-14 h-20';
  const textLg = small ? 'text-xs' : 'text-base';
  const textSm = small ? 'text-[9px]' : 'text-xs';

  if (faceDown) {
    return (
      <div
        className={`${w} rounded-md flex-shrink-0 card-flip`}
        style={{
          background: 'linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%)',
          border: '1px solid #3b82f6',
          animationDelay: `${animDelay}ms`,
          boxSizing: 'border-box',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {/* diamond pattern */}
        <div style={{ opacity: 0.3, fontSize: small ? 10 : 14 }}>🂠</div>
      </div>
    );
  }

  const { rank, symbol } = parseCard(card);
  const suitColor = SUIT_COLORS[symbol] ?? '#1f2937';
  const isRed = suitColor === '#dc2626';

  return (
    <div
      className={`${w} rounded-md flex-shrink-0 card-flip select-none`}
      style={{
        background: highlight ? '#fef3c7' : '#f8f8eb',
        border: highlight ? '1.5px solid #d4af37' : '1px solid #ccc',
        boxShadow: highlight
          ? '0 0 0 2px rgba(212,175,55,0.4), 0 2px 6px rgba(0,0,0,0.3)'
          : '0 2px 6px rgba(0,0,0,0.3)',
        animationDelay: `${animDelay}ms`,
        color: suitColor,
        display: 'flex',
        flexDirection: 'column',
        padding: small ? '2px 3px' : '3px 4px',
        position: 'relative',
        boxSizing: 'border-box',
      }}
    >
      {/* Top-left rank + suit */}
      <div style={{ lineHeight: 1 }}>
        <div className={`${textSm} font-bold font-mono leading-none`}>{rank}</div>
        <div className={`${textSm} leading-none`}>{symbol}</div>
      </div>
      {/* Center large suit */}
      <div
        className={`absolute inset-0 flex items-center justify-center ${textLg} font-bold`}
        style={{ color: suitColor, opacity: isRed ? 0.85 : 0.75 }}
      >
        {symbol}
      </div>
    </div>
  );
}
