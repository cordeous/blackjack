import { SessionProvider, useSession } from './context/SessionContext';
import { MainMenuView }    from './views/MainMenuView';
import { GameView }        from './views/GameView';
import { LeaderboardView } from './views/LeaderboardView';

function LoadingView() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6">
      <div className="text-6xl animate-bounce select-none">♠</div>
      <div className="text-xl font-semibold text-yellow-400">Simulating Tournament…</div>
      <div className="text-sm text-green-400">Running all agents against the dealer</div>
      <div className="w-10 h-10 border-4 border-yellow-400/30 border-t-yellow-400 rounded-full animate-spin mt-2" />
    </div>
  );
}

function AppInner() {
  const { state } = useSession();

  return (
    <div className="min-h-screen" style={{ background: '#0a2212' }}>
      {state.view === 'menu'         && <MainMenuView />}
      {state.view === 'loading'      && <LoadingView />}
      {state.view === 'game'         && state.session && <GameView />}
      {state.view === 'leaderboard'  && state.session && <LeaderboardView />}
    </div>
  );
}

export default function App() {
  return (
    <SessionProvider>
      <AppInner />
    </SessionProvider>
  );
}
