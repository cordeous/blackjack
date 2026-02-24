import { useState, useCallback, useMemo } from 'react';
import type { SessionResult, DisplayEvent } from '../types/session';

// ---------------------------------------------------------------------------
// Build flat event list from session data
// ---------------------------------------------------------------------------

export function buildEventList(session: SessionResult): DisplayEvent[] {
  const events: DisplayEvent[] = [];
  session.rounds.forEach((round, roundIndex) => {
    events.push({ kind: 'round_start', roundIndex });

    round.agent_results.forEach((ar, agentIndex) => {
      ar.steps.forEach((_step, stepIndex) => {
        events.push({ kind: 'agent_step', roundIndex, agentIndex, stepIndex });
      });
    });

    events.push({ kind: 'dealer_reveal', roundIndex });
    events.push({ kind: 'round_end',    roundIndex });
  });
  return events;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export type ReplaySpeed = 400 | 800 | 1600 | 3200;

export interface ReplayState {
  eventIndex:   number;
  totalEvents:  number;
  currentEvent: DisplayEvent;
  isPlaying:    boolean;
  speed:        ReplaySpeed;
  atEnd:        boolean;
}

export interface ReplayControls {
  play():               void;
  pause():              void;
  next():               void;
  prev():               void;
  jumpToEnd():          void;
  jumpTo(idx: number):  void;
  setSpeed(s: ReplaySpeed): void;
}

export function useReplay(session: SessionResult): [ReplayState, ReplayControls] {
  const events = useMemo(() => buildEventList(session), [session]);

  const [eventIndex, setEventIndex] = useState(0);
  const [isPlaying, setIsPlaying]   = useState(true);
  const [speed, setSpeedState]      = useState<ReplaySpeed>(800);

  const totalEvents  = events.length;
  const currentEvent = events[Math.min(eventIndex, totalEvents - 1)];
  const atEnd        = eventIndex >= totalEvents - 1;

  const next = useCallback(() => {
    setEventIndex(i => Math.min(i + 1, totalEvents - 1));
  }, [totalEvents]);

  const prev = useCallback(() => {
    setEventIndex(i => Math.max(i - 1, 0));
  }, []);

  const play = useCallback(() => {
    if (eventIndex >= totalEvents - 1) setEventIndex(0);
    setIsPlaying(true);
  }, [eventIndex, totalEvents]);

  const pause     = useCallback(() => setIsPlaying(false), []);
  const jumpToEnd = useCallback(() => { setEventIndex(totalEvents - 1); setIsPlaying(false); }, [totalEvents]);
  const jumpTo    = useCallback((idx: number) => setEventIndex(Math.max(0, Math.min(idx, totalEvents - 1))), [totalEvents]);
  const setSpeed  = useCallback((s: ReplaySpeed) => setSpeedState(s), []);

  const replayState: ReplayState = { eventIndex, totalEvents, currentEvent, isPlaying, speed, atEnd };
  const controls: ReplayControls = { play, pause, next, prev, jumpToEnd, jumpTo, setSpeed };

  return [replayState, controls];
}
