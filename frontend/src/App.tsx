import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { GameState } from "./types";
import "./styles.css";

type Role = "P1" | "P2";
type GameMode = "PVP" | "PVAI";

function getOrCreateLocalToken(key: string) {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const token = crypto.randomUUID();
  localStorage.setItem(key, token);
  return token;
}

const PIECE_MAP: Record<string, string> = {
  P: "/pieces/wP.svg",
  N: "/pieces/wN.svg",
  B: "/pieces/wB.svg",
  R: "/pieces/wR.svg",
  Q: "/pieces/wQ.svg",
  K: "/pieces/wK.svg",
  p: "/pieces/bP.svg",
  n: "/pieces/bN.svg",
  b: "/pieces/bB.svg",
  r: "/pieces/bR.svg",
  q: "/pieces/bQ.svg",
  k: "/pieces/bK.svg",
  ".": "",
};

export default function App() {
  const [gameMode, setGameMode] = useState<GameMode>("PVP");

  const [gameId, setGameId] = useState<string>("");
  const [joinCode, setJoinCode] = useState<string>("");
  const [playerToken, setPlayerToken] = useState<string>("");
  const [role, setRole] = useState<Role | "">("");
  const [state, setState] = useState<GameState | null>(null);
  const [error, setError] = useState<string>("");
  const [selected, setSelected] = useState<{ r: number; c: number } | null>(null);
  const [lastMove, setLastMove] = useState<{ from: { r: number; c: number }; to: { r: number; c: number } } | null>(null);
  const [copied, setCopied] = useState(false);

  useMemo(() => getOrCreateLocalToken("chess_client_id"), []);

  async function onCreate() {
    try {
      const res = await api.createGame(gameMode);
      setGameId(res.gameId);
      setJoinCode(res.joinCode ?? "");
      setState(res.state);
      setRole(res.player);
      setPlayerToken(res.playerToken);
      setSelected(null);
      setLastMove(null);
      setError("");
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function onJoin() {
    try {
      const res = await api.joinGame(joinCode.trim());
      setGameId(res.gameId);
      setRole(res.player);
      setPlayerToken(res.playerToken);
      setState(res.state);
      setSelected(null);
      setLastMove(null);
      setError("");
      setGameMode("PVP");
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function refresh() {
    if (!gameId) {
      setError("No active game to refresh");
      return;
    }
    try {
      const res = await api.getGame(gameId);
      setState(res.state);
      setError("");
    } catch (e: any) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (!gameId) return;
    const t = setInterval(refresh, 1200);
    return () => clearInterval(t);
  }, [gameId]);

  async function submitMove(from: { r: number; c: number }, to: { r: number; c: number }) {
    if (!gameId || !playerToken || !state) return;

    if (state.status !== "ACTIVE") {
      setError("Game is not active");
      setSelected(null);
      return;
    }

    if ((state.turn === "P1" && role !== "P1") || (state.turn === "P2" && role !== "P2")) {
      setError("Not your turn");
      setSelected(null);
      return;
    }

    try {
      const res = await api.makeMove(gameId, playerToken, { from, to });
      setState(res.state);
      setLastMove({ from, to });
      setSelected(null);
      setError("");

      if (gameMode === "PVAI" && role === "P1" && res.state.status === "ACTIVE" && res.state.turn === "P2") {
        setTimeout(async () => {
          try {
            const botRes = await api.aiMove(gameId, "medium");
            setState(botRes.state);
          } catch (e: any) {
            setError(e.message);
          }
        }, 700);
      } 
    } catch (e: any) {
      setError(e.message);
      setSelected(null);
    }
  }

  async function requestAiMove() {
    if (!gameId || !state) return;
  
    if (state.status !== "ACTIVE") {
      setError("Game is not active");
      return;
    }
  
    if ((state.turn === "P1" && role !== "P1") || (state.turn === "P2" && role !== "P2")) {
      setError("Only the player whose turn it is can request the AI move");
      return;
    }
  
    setError("");
    try {
      const res = await api.aiMove(gameId, "medium", playerToken);
      setState(res.state);
  
      if (gameMode === "PVAI" && role === "P1" && res.state.status === "ACTIVE" && res.state.turn === "P2") {
        setTimeout(async () => {
          try {
            const botRes = await api.aiMove(gameId, "medium");
            setState(botRes.state);
          } catch (e: any) {
            setError(e.message);
          }
        }, 700);
      }
    } catch (e: any) {
      setError(e.message);
    }
  }
  async function copyJoinCode() {
    if (!joinCode) return;
    try {
      await navigator.clipboard.writeText(joinCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError("Failed to copy join code");
    }
  }

  function isLastMove(r: number, c: number) {
    if (!lastMove) return false;
    return (
      (lastMove.from.r === r && lastMove.from.c === c) ||
      (lastMove.to.r === r && lastMove.to.c === c)
    );
  }

  const isBlackView = role === "P2";

  const roleLabel =
    role === "P1"
      ? "You are White (P1)"
      : role === "P2"
      ? "You are Black (P2)"
      : "Not joined yet";

  const topPlayer = isBlackView
    ? { label: "White (P1)", isYou: role === "P1", avatar: "♔", colorClass: "white" }
    : { label: "Black (P2)", isYou: role === "P2", avatar: "♚", colorClass: "black" };

  const bottomPlayer = isBlackView
    ? { label: "Black (P2)", isYou: role === "P2", avatar: "♚", colorClass: "black" }
    : { label: "White (P1)", isYou: role === "P1", avatar: "♔", colorClass: "white" };

  const canActOnTurn =
    !!state &&
    state.status === "ACTIVE" &&
    ((state.turn === "P1" && role === "P1") || (state.turn === "P2" && role === "P2"));

  return (
    <div className="page">
      <header className="header">
        <div className="header-logo">♟</div>
        <div>
          <h1>AI Chess Arena</h1>
        </div>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Game Setup</h2>

          <div className="row">
            <button
              className={gameMode === "PVP" ? "primary" : ""}
              onClick={() => setGameMode("PVP")}
              style={{ flex: 1 }}
            >
              vs Player
            </button>
            <button
              className={gameMode === "PVAI" ? "primary" : ""}
              onClick={() => setGameMode("PVAI")}
              style={{ flex: 1 }}
            >
              vs AI
            </button>
          </div>

          <div className="row">
            <button className="primary" onClick={onCreate} style={{ flex: 1 }}>
              + New Game
            </button>
          </div>

          {gameMode === "PVP" && (
            <>
              <div className="divider" />

              <div className="row">
                <label>Join Code</label>
                <input
                  value={joinCode}
                  onChange={(e) => setJoinCode(e.target.value)}
                  placeholder="Enter join code"
                />
              </div>

              <div className="row">
                <button onClick={onJoin} style={{ flex: 1 }}>
                  Join as Black (P2)
                </button>
              </div>

              {joinCode && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                    Share this join code
                  </div>

                  <div className="code-display">
                    <span>🔑 Join Code:</span>
                    <span>{joinCode}</span>
                  </div>

                  <button
                    onClick={copyJoinCode}
                    className="primary"
                    style={{ marginTop: 6, width: "100%" }}
                  >
                    {copied ? "Copied!" : "Copy Join Code"}
                  </button>
                </div>
              )}
            </>
          )}

          {gameMode === "PVAI" && (
            <>
              <div className="divider" />
              <div className="muted" style={{ textAlign: "left" }}>
                In this mode, you play as White (P1) and the bot plays as Black (P2).
                The bot will move automatically after your move.
              </div>
            </>
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
                <div>
                  <b>Turn</b> {state.turn === "P1" ? "White (P1)" : "Black (P2)"}
                </div>
                <div>
                  <b>Role</b> {roleLabel}
                </div>
                <div>
                  <b>Mode</b> {gameMode === "PVP" ? "Player vs Player" : "Player vs AI"}
                </div>
                {state.message && (
                  <div>
                    <b>Message</b> {state.message}
                  </div>
                )}
              </div>
            </>
          )}
        </section>

        <section className="card">
          <h2>Board</h2>

          {!state ? (
            <div className="muted" style={{ padding: "40px 0", fontSize: 14 }}>
              Create or join a game to start playing.
            </div>
          ) : (
            <div className="board-wrapper">
              <div className="player-bar">
                <div className={`player-avatar ${topPlayer.colorClass}`}>{topPlayer.avatar}</div>
                <span className="player-name">{topPlayer.label}</span>
                <span className="player-role">{topPlayer.isYou ? "You" : "Opponent"}</span>
              </div>

              <div className="board">
                {Array.from({ length: 8 }).map((_, r) => (
                  <div className="boardRow" key={r}>
                    {Array.from({ length: 8 }).map((_, c) => {
                      const boardR = isBlackView ? 7 - r : r;
                      const boardC = isBlackView ? 7 - c : c;

                      const idx = boardR * 8 + boardC;
                      const cell = state.board[idx] ?? ".";
                      const isEmpty = cell === ".";
                      const isSelected = selected?.r === boardR && selected?.c === boardC;

                      function onCellClick() {
                        if (!selected) {
                          if (!isEmpty) setSelected({ r: boardR, c: boardC });
                        } else {
                          submitMove(selected, { r: boardR, c: boardC });
                        }
                      }

                      return (
                        <div
                          key={`${r}-${c}`}
                          className={[
                            "cell",
                            isSelected ? "selected" : "",
                            isLastMove(boardR, boardC) ? "last-move" : "",
                          ].join(" ")}
                          onClick={onCellClick}
                        >
                          {!isEmpty && (
                            <img
                              src={PIECE_MAP[cell]}
                              alt={cell}
                              className="piece-img"
                              draggable={false}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>

              <div className="player-bar">
                <div className={`player-avatar ${bottomPlayer.colorClass}`}>{bottomPlayer.avatar}</div>
                <span className="player-name">{bottomPlayer.label}</span>
                <span className="player-role">{bottomPlayer.isYou ? "You" : "Opponent"}</span>
              </div>

              <div className="board-actions">
                <button className="primary" onClick={requestAiMove} disabled={!canActOnTurn}>
                  🤖 AI Move
                </button>
              </div>

              <div className="muted">{roleLabel}. Click a piece, then click a destination square.</div>
            </div>
          )}
        </section>
      </div>

      <footer className="footer">
        <span>API: {import.meta.env.VITE_API_BASE ?? "http://localhost:8080"}</span>
        <span>Playing as: {role === "P1" ? "White (P1)" : role === "P2" ? "Black (P2)" : "—"}</span>
      </footer>
    </div>
  );
}