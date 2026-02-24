import { CardDisplay } from './CardDisplay';

interface Props {
  cards:      string[];
  small?:     boolean;
  highlight?: boolean;
  showValue?: boolean;
  value?:     number;
  isBust?:    boolean;
  isBlackjack?: boolean;
}

export function HandDisplay({ cards, small = false, highlight = false, showValue = false, value, isBust, isBlackjack }: Props) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex flex-wrap justify-center gap-1">
        {cards.map((card, i) => (
          <CardDisplay
            key={`${card}-${i}`}
            card={card}
            small={small}
            highlight={highlight}
            animDelay={i * 60}
          />
        ))}
      </div>
      {showValue && value !== undefined && (
        <div
          className="text-xs font-bold px-2 py-0.5 rounded-full"
          style={{
            backgroundColor: isBust ? '#dc2626' : isBlackjack ? '#d4af37' : 'rgba(255,255,255,0.15)',
            color: isBust ? 'white' : isBlackjack ? '#0c1a08' : 'white',
          }}
        >
          {isBust ? `${value} BUST` : isBlackjack ? `${value} BJ!` : value}
        </div>
      )}
    </div>
  );
}
