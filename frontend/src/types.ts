export type Player = "P1" | "P2";

export type GameState = {
  board: string[];
  fen?: string;
  turn: Player;
  status: "ACTIVE" | "DONE";
  message: string;
  lastMoveSan?: string | null;
  lastMoveUci?: string | null;
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