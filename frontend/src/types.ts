export type Player = "P1" | "P2";

export type BlunderTopMove = {
  move: string;
  score: number;
};

export type BlunderInfo = {
  fen: string;
  played_move: string;
  best_move: string;
  played_score: number;
  best_score: number;
  drop: number;
  played_rank: number;
  total_legal_moves: number;
  rank_pct: number;
  is_blunder: boolean;
  severity: string;
  threshold: number;
  top_moves: BlunderTopMove[];
};

export type GameState = {
  board: string[];
  fen?: string;
  turn: Player;
  status: "ACTIVE" | "DONE";
  message: string;
  lastMoveSan?: string | null;
  lastMoveUci?: string | null;
  blunderByPlayer?: Partial<Record<Player, BlunderInfo | null>>;
};

export type CreateGameResponse = {
  gameId: string;
  joinCode: string;
  player: Player;
  playerToken: string;
  state: GameState;
};

export type JoinResponse = {
  gameId: string;
  player: Player;
  playerToken: string;
  state: GameState;
};

export type GetGameResponse = {
  gameId: string;
  createdAt?: string;
  updatedAt?: string;
  state: GameState;
};

export type MoveResponse = {
  gameId: string;
  moveId: string;
  state: GameState;
};