// ---------------------------------------------------------------------------
// Session data types — mirror the FastAPI JSON response from /run
// ---------------------------------------------------------------------------

export interface AgentRoundStep {
  agent_name:    string;
  agent_index:   number;
  player_hand:   string[];    // e.g. ["5♥", "K♣"]
  dealer_upcard: string;      // e.g. "A♠"
  legal_actions: string[];
  action_taken:  string;      // "hit" | "stand" | "double" | "split"
  reason:        string;
  hand_value:    number;
  is_split_hand: boolean;
  hand_index:    number;
}

export interface AgentRoundResult {
  agent_name:    string;
  agent_index:   number;
  player_hands:  string[][];   // may have 2 after split
  bets:          number[];
  payouts:       number[];
  actions_taken: string[][];
  bankroll_after: number;
  net_payout:    number;
  steps:         AgentRoundStep[];
}

export interface Round {
  round_num:     number;
  dealer_hand:   string[];
  dealer_upcard: string;
  agent_results: AgentRoundResult[];
}

export interface LeaderboardEntry {
  name:              string;
  agent_index:       number;
  starting_bankroll: number;
  final_bankroll:    number;
  wins:              number;
  losses:            number;
  ties:              number;
  blackjacks:        number;
  total_payout:      number;
  points:            number;
  win_rate:          number;
  net_profit:        number;
  rounds_played:     number;
}

export interface SessionResult {
  rounds_played:     number;
  starting_bankroll: number;
  rounds:            Round[];
  leaderboard:       LeaderboardEntry[];
}

// ---------------------------------------------------------------------------
// Replay event types
// ---------------------------------------------------------------------------

export type DisplayEventKind =
  | 'round_start'
  | 'agent_step'
  | 'dealer_reveal'
  | 'round_end';

export interface DisplayEvent {
  kind:        DisplayEventKind;
  roundIndex:  number;   // index into session.rounds[]
  agentIndex?: number;   // for agent_step
  stepIndex?:  number;   // index into agent.steps[]
}

// ---------------------------------------------------------------------------
// Config submitted to /run
// ---------------------------------------------------------------------------

export interface RunConfig {
  agents:            string[];
  num_rounds:        number;
  base_bet:          number;
  starting_bankroll: number;
  seed:              number | null;
  mcts_sims:         number;
  dnn_model_path:    string;
}

// ---------------------------------------------------------------------------
// Agent colour palette (matches backend agent order)
// ---------------------------------------------------------------------------

export const AGENT_COLORS: Record<string, string> = {
  Random:                  '#f59e0b',   // amber
  'Heuristic(basic)':      '#38bdf8',   // sky blue
  'Heuristic(aggressive)': '#fb923c',   // orange
  MCTS:                    '#a78bfa',   // violet
  DNN:                     '#34d399',   // emerald
};

export const ACTION_COLORS: Record<string, string> = {
  hit:    '#16a34a',
  stand:  '#dc2626',
  double: '#d97706',
  split:  '#7c3aed',
};

export const SUIT_COLORS: Record<string, string> = {
  '♥': '#dc2626',
  '♦': '#dc2626',
  '♣': '#1f2937',
  '♠': '#1f2937',
};
