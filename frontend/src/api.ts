import type { CreateGameResponse, GetGameResponse, JoinResponse, MoveResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }

  // Some endpoints may return empty — but ours return json
  return (await res.json()) as T;
}

export const api = {
  createGame: () => http<CreateGameResponse>("/games", { method: "POST", body: "{}" }),
  joinGame: (gameId: string, joinCode: string) =>
    http<JoinResponse>(`/games/${gameId}/join`, { method: "POST", body: JSON.stringify({ joinCode }) }),
  getGame: (gameId: string) => http<GetGameResponse>(`/games/${gameId}`),
  makeMove: (gameId: string, playerToken: string, move: any) =>
    http<MoveResponse>(`/games/${gameId}/move`, {
      method: "POST",
      body: JSON.stringify({ playerToken, move, clientTs: new Date().toISOString() }),
    }),
  aiMove: (gameId: string, difficulty: string) =>
    http<MoveResponse>(`/games/${gameId}/ai-move`, { method: "POST", body: JSON.stringify({ difficulty }) }),
};
