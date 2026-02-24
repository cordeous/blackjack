import { useEffect, useRef } from 'react';

/**
 * Calls `onTick` every `intervalMs` when `active` is true.
 * Clears the interval when `active` becomes false or component unmounts.
 */
export function useAutoPlay(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): void {
  const tickRef = useRef(onTick);
  tickRef.current = onTick;

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => tickRef.current(), intervalMs);
    return () => clearInterval(id);
  }, [active, intervalMs]);
}
