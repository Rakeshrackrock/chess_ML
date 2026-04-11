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

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["8", "7", "6", "5", "4", "3", "2", "1"];

const PIECE_MAP: Record<string, string> = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
  ".": "",
};

export default function App() {
  const [gameId, setGameId] = useState<string>("");
  const [joinCode, setJoinCode] = useState<string>("");
  const [playerToken, setPlayerToken] = useState<string>("");
  const [role, setRole] = useState<Role | "">("");
  const [state, setState] = useState<GameState | null>(null);
  const [error, setError] = useState<string>("");
  const [selected, setSelected] = useState<{ r: number; c: number } | null>(null);
  const [lastMove, setLastMove] = useState<{ from: {r:number,c:number}, to: {r:number,c:number} } | null>(null);

  useMemo(() => getOrCreateLocalToken("shatigo_client_id"), []);

  async function onCreate() {
    try {
      const res = await api.createGame();
      setGameId(res.gameId);
      setJoinCode(res.joinCode);
      setState(res.state);
      setRole(res.player);
      setPlayerToken(res.playerToken);
      setError("");
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function onJoin() {
    try {
      const res = await api.joinGame(gameId.trim(), joinCode.trim());
      setRole(res.player);
      setPlayerToken(res.playerToken);
      setState(res.state);
      setError("");
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

  async function submitMove(from: { r: number; c: number }, to: { r: number; c: number }) {
    if (!gameId || !playerToken) return;
    try {
      const res = await api.makeMove(gameId, playerToken, { from, to });
      setState(res.state);
      setLastMove({ from, to });
      setError("");
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

  function isLastMove(r: number, c: number) {
    if (!lastMove) return false;
    return (lastMove.from.r === r && lastMove.from.c === c) ||
           (lastMove.to.r   === r && lastMove.to.c   === c);
  }

  console.log("STATE:", state);
  console.log("BOARD:", state?.board);

  //const turnIsWhite = state?.turn === "white" || state?.turn === "P1" || state?.turn === "w";

  return (
    <div className="page">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">♟</div>
        <div>
          <h1>Chess on GCP</h1>
          <p className="sub">No login · Cloud Run + Firestore · ML stub ready</p>
        </div>
      </header>

      <div className="grid">
        {/* ── Setup Panel ── */}
        <section className="card">
          <h2>Game Setup</h2>

          <div className="row">
            <button className="primary" onClick={onCreate} style={{ flex: 1 }}>
              + New Game
            </button>
          </div>

          <div className="divider" />

          <div className="row">
            <label>Game ID</label>
            <input
              value={gameId}
              onChange={(e) => setGameId(e.target.value)}
              placeholder="game_..."
            />
          </div>

          <div className="row">
            <label>Join Code</label>
            <input
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value)}
              placeholder="e.g. a1b2c3"
            />
          </div>

          <div className="row">
            <button onClick={onJoin} style={{ flex: 1 }}>Join as Black</button>
            <button onClick={refresh}>↻ Refresh</button>
          </div>

          {joinCode && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Share join code</div>
              <div className="code-display">
                <span>🔗</span>
                <span>{joinCode}</span>
              </div>
            </div>
          )}

          {error && <div className="error">⚠ {error}</div>}

          {state && (
            <>
              <div className="divider" />
              <div className="kv">
                <div>
                  <b>Status</b>
                  <span className="status-pill">{state.status}</span>
                </div>
                <div><b>Turn</b> {state.turn}</div>
                {state.message && <div><b>Message</b> {state.message}</div>}
              </div>
            </>
          )}
        </section>

        {/* ── Board Panel ── */}
        <section className="card">
          <h2>Board</h2>

          {!state ? (
            <div className="muted" style={{ padding: '40px 0', fontSize: 14 }}>
              Create or join a game to start playing.
            </div>
          ) : (
            <div className="board-wrapper">
              {/* Black player bar */}
              <div className="player-bar">
                <div className="player-avatar black">♚</div>
                <span className="player-name">Black</span>
                <span className="player-role">{role === "P2" ? "You" : "Opponent"}</span>
              </div>

              {/* Board with coordinate labels */}
              <div className="board-coords-wrap">
                <div className="coords-ranks">
                  {RANKS.map(r => (
                    <div className="coord-rank" key={r}>{r}</div>
                  ))}
                </div>
                <div>
                  <div className="board">
                    {Array.from({ length: 8 }).map((_, r) => (
                      <div className="boardRow" key={r}>
                        {Array.from({ length: 8 }).map((_, c) => {
                          const idx = r * 8 + c;
                          const cell = state.board[idx] ?? ".";
                          const isEmpty = cell === "." || cell === " " || !cell;
                          const isSelected = selected?.r === r && selected?.c === c;

                          function onCellClick() {
                            if (!selected) {
                              if (!isEmpty) setSelected({ r, c });
                            } else {
                              submitMove(selected, { r, c });
                              setSelected(null);
                            }
                          }

                          return (
                            <div
                              key={`${r}-${c}`}
                              className={[
                                "cell",
                                isSelected ? "selected" : "",
                                isLastMove(r, c) ? "last-move" : "",
                              ].join(" ")}
                              onClick={onCellClick}
                            >
                              {!isEmpty && <span className="piece">{PIECE_MAP[cell] ?? ""}</span>}
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                  {/* File labels */}
                  <div style={{ display: 'flex', paddingLeft: 0, marginTop: 3 }}>
                    {FILES.map(f => (
                      <div className="coord-file" key={f}>{f}</div>
                    ))}
                  </div>
                </div>
              </div>

              {/* White player bar */}
              <div className="player-bar">
                <div className="player-avatar white">♔</div>
                <span className="player-name">White</span>
                <span className="player-role">{role === "P1" ? "You" : "Opponent"}</span>
              </div>

              {/* Turn indicator */}
              {/* <div className="turn-bar">
                <div className={`turn-dot ${turnIsWhite ? "white" : "black"}`} />
                <span>{turnIsWhite ? "White to move" : "Black to move"}</span>
              </div> */}

              {/* Actions */}
              <div className="board-actions">
                <button className="primary" onClick={requestAiMove}>🤖 AI Move</button>
              </div>

              <div className="muted">Click a piece, then click a destination square.</div>
            </div>
          )}
        </section>
      </div>

      <footer className="footer">
        <span>API: {import.meta.env.VITE_API_BASE ?? "http://localhost:8080"}</span>
        <span>Playing as: {role || "—"}</span>
      </footer>
    </div>
  );
}
