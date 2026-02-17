export type Player = "P1" | "P2";

export type GameState = {
  board: string[]; // length 64, flattened
  turn: Player;
  status: "ACTIVE" | "DONE";
  message: string;
};

export type CreateGameResponse = {
  gameId: string;
  joinCode: string;
  player: Player;
  playerToken: string; // NOW returned
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
