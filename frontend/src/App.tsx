import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { GameState } from "./types";
import "./styles.css";

type Role = "P1" | "P2";

function getOrCreateLocalToken(key: string) {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const token = crypto.randomUUID();
  localStorage.setItem(key, token);
  return token;
}

export default function App() {
  const [gameId, setGameId] = useState<string>("");
  const [joinCode, setJoinCode] = useState<string>("");
  const [playerToken, setPlayerToken] = useState<string>(""); // backend token
  const [role, setRole] = useState<Role | "">("");
  const [state, setState] = useState<GameState | null>(null);
  const [error, setError] = useState<string>("");

  useMemo(() => getOrCreateLocalToken("shatigo_client_id"), []);


  async function onCreate() {
    setError("");
    try {
      const res = await api.createGame();
      setGameId(res.gameId);
      setJoinCode(res.joinCode);
      setState(res.state);
      setRole(res.player);
      setPlayerToken(res.playerToken);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function onJoin() {
    setError("");
    try {
      const res = await api.joinGame(gameId.trim(), joinCode.trim());
      setRole(res.player);
      setPlayerToken(res.playerToken);
      setState(res.state);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function refresh() {
    if (!gameId) return;
    try {
      const res = await api.getGame(gameId);
      setState(res.state);
    } catch {}
  }

  useEffect(() => {
    if (!gameId) return;
    const t = setInterval(refresh, 1200);
    return () => clearInterval(t);
  }, [gameId]);

  async function submitDummyMove() {
    if (!gameId) return;
    setError("");

    if (!playerToken) {
      setError("You don't have a player token yet (join from another browser/tab).");
      return;
    }

    try {
      const move = { from: { r: 6, c: 0 }, to: { r: 5, c: 0 } };
      const res = await api.makeMove(gameId, playerToken, move);
      setState(res.state);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function requestAiMove() {
    if (!gameId) return;
    setError("");
    try {
      const res = await api.aiMove(gameId, "medium");
      setState(res.state);
    } catch (e: any) {
      setError(e.message);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Shatigo on GCP</h1>
          <p className="sub">No login • Cloud Run + Firestore • ML stub ready</p>
        </div>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Game Setup</h2>

          <div className="row">
            <button onClick={onCreate}>Create Game</button>
          </div>

          <div className="row">
            <label>Game ID</label>
            <input value={gameId} onChange={(e) => setGameId(e.target.value)} placeholder="game_..." />
          </div>

          <div className="row">
            <label>Join Code</label>
            <input value={joinCode} onChange={(e) => setJoinCode(e.target.value)} placeholder="e.g. a1b2c3" />
          </div>

          <div className="row">
            <button onClick={onJoin}>Join as P2</button>
            <button onClick={refresh}>Refresh</button>
          </div>

          {error && <div className="error">{error}</div>}

          {state && (
            <div className="kv">
              <div><b>Status:</b> {state.status}</div>
              <div><b>Turn:</b> {state.turn}</div>
              <div><b>Message:</b> {state.message}</div>
            </div>
          )}

          <div className="hint">
            Tip: open this page in two browsers. Create in one, join in the other.
          </div>
        </section>

        <section className="card">
          <h2>Board (placeholder)</h2>

          {!state ? (
            <div className="muted">Create or join a game to see the board.</div>
          ) : (
            <div className="board">
              {Array.from({ length: 8 }).map((_, r) => (
                <div className="boardRow" key={r}>
                  {Array.from({ length: 8 }).map((_, c) => {
                    const idx = r * 8 + c;
                    const cell = state.board[idx] ?? ".";
                    return (
                      <div className="cell" key={`${r}-${c}`}>
                        {cell}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          )}

          <div className="row">
            <button onClick={submitDummyMove}>Submit Dummy Move</button>
            <button onClick={requestAiMove}>AI Move (stub)</button>
          </div>

          <div className="muted">
            Next: replace the placeholder board with real Shatigo UI + legal moves.
          </div>
        </section>
      </div>

      <footer className="footer">
        <span>API: {import.meta.env.VITE_API_BASE ?? "http://localhost:8080"}</span>
        <span>Role: {role || "-"}</span>
      </footer>
    </div>
  );
}
