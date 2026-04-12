from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
import secrets
import chess
from typing import Any, Dict, Optional, Literal
import os
import torch
import time

# Uncomment for GCP deployment
from google.cloud import datastore

from agent import load_model, predict_best_move

app = FastAPI(title="Chess Game API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://shatigoai.web.app",
        "https://shatigoai.firebaseapp.com",
        "https://chessai-e0c8c.web.app",
        "https://chessai-e0c8c.firebaseapp.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Storage mode
# -----------------------------------
# Local now, switch to False later on GCP
USE_LOCAL_STORE = False

# Uncomment for GCP deployment
ds = datastore.Client()

LOCAL_GAMES: dict[str, dict] = {}
LOCAL_MOVES: dict[str, dict] = {}

GAMES_KIND = "Game"
MOVES_KIND = "Move"

# -----------------------------------
# Model loading
# -----------------------------------
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "latest.pt")
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_CHECKPOINT):
            raise HTTPException(500, f"Checkpoint not found: {MODEL_CHECKPOINT}")
        MODEL = load_model(MODEL_CHECKPOINT, MODEL_DEVICE)
    return MODEL


# -----------------------------------
# Helpers
# -----------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_public_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(10)}"


def fen_to_flat_board(fen: str) -> list[str]:
    """
    Converts a FEN board into the frontend's expected 64-cell row-major array.
    Top-left is a8, bottom-right is h1.
    """
    board = chess.Board(fen)
    cells: list[str] = []

    for rank in range(7, -1, -1):  # 8 -> 1
        for file in range(8):      # a -> h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            cells.append(piece.symbol() if piece else ".")

    return cells


def rc_to_square(r: int, c: int) -> chess.Square:
    """
    Frontend coordinates:
      r=0,c=0 => a8
      r=7,c=7 => h1
    """
    if not (0 <= r <= 7 and 0 <= c <= 7):
        raise HTTPException(400, f"Invalid board coordinates: r={r}, c={c}")

    file_idx = c
    rank_idx = 7 - r
    return chess.square(file_idx, rank_idx)


def move_dict_to_chess_move(move: Dict[str, Any], board: chess.Board) -> chess.Move:
    """
    Converts frontend move payload:
      { "from": {"r": 6, "c": 4}, "to": {"r": 4, "c": 4} }
    into a python-chess Move.
    Auto-promotes pawns to queen.
    """
    try:
        from_r = int(move["from"]["r"])
        from_c = int(move["from"]["c"])
        to_r = int(move["to"]["r"])
        to_c = int(move["to"]["c"])
    except Exception:
        raise HTTPException(400, "Move must contain from/to with r/c integers")

    from_sq = rc_to_square(from_r, from_c)
    to_sq = rc_to_square(to_r, to_c)

    candidate = chess.Move(from_sq, to_sq)

    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and to_rank == 7) or (
            piece.color == chess.BLACK and to_rank == 0
        ):
            candidate = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return candidate


def game_status_from_board(board: chess.Board) -> str:
    return "DONE" if board.is_game_over() else "ACTIVE"


def game_message_from_board(board: chess.Board, last_actor: str, last_san: str) -> str:
    if board.is_checkmate():
        return f"{last_san} — checkmate. {last_actor} wins."
    if board.is_stalemate():
        return f"{last_san} — stalemate."
    if board.is_insufficient_material():
        return f"{last_san} — draw by insufficient material."
    if board.is_seventyfive_moves():
        return f"{last_san} — draw by 75-move rule."
    if board.is_fivefold_repetition():
        return f"{last_san} — draw by fivefold repetition."
    if board.is_check():
        return f"{last_actor} played {last_san} (check)"
    return f"{last_actor} played {last_san}"


def initial_state() -> Dict[str, Any]:
    board = chess.Board()
    fen = board.fen()
    return {
        "fen": fen,
        "board": fen_to_flat_board(fen),
        "turn": "P1",   # White
        "status": "ACTIVE",
        "message": "Game created",
        "lastMoveSan": None,
        "lastMoveUci": None,
    }


# -----------------------------------
# Request / Response models
# -----------------------------------
class CreateGameRequest(BaseModel):
    mode: Literal["PVP", "PVAI"] = "PVP"


class CreateGameResponse(BaseModel):
    gameId: str
    joinCode: Optional[str] = None
    player: str
    playerToken: str
    state: Dict[str, Any]


class JoinRequest(BaseModel):
    joinCode: str


class JoinResponse(BaseModel):
    gameId: str
    player: str
    playerToken: str
    state: Dict[str, Any]


class MoveRequest(BaseModel):
    playerToken: str
    move: Dict[str, Any]
    clientTs: Optional[str] = None


class MoveResponse(BaseModel):
    gameId: str
    state: Dict[str, Any]
    moveId: str


class AiMoveRequest(BaseModel):
    difficulty: Optional[str] = "medium"
    playerToken: Optional[str] = None


# -----------------------------------
# Storage abstraction
# -----------------------------------
def get_game_entity(game_id: str):
    if USE_LOCAL_STORE:
        entity = LOCAL_GAMES.get(game_id)
        if not entity:
            raise HTTPException(404, "Game not found")
        return entity

    # Uncomment for GCP deployment
    entity = ds.get(ds.key(GAMES_KIND, game_id))
    if not entity:
        raise HTTPException(404, "Game not found")
    return dict(entity)

    # raise HTTPException(500, "Datastore mode not enabled")


def find_game_by_join_code(join_code: str):
    if USE_LOCAL_STORE:
        for game in LOCAL_GAMES.values():
            if game.get("joinCode") == join_code:
                return game
        raise HTTPException(404, "Game not found for this join code")

    # Uncomment for GCP deployment
    query = ds.query(kind=GAMES_KIND)
    query.add_filter("joinCode", "=", join_code)
    results = list(query.fetch(limit=1))
    if not results:
        raise HTTPException(404, "Game not found for this join code")
    return dict(results[0])

    # raise HTTPException(500, "Datastore mode not enabled")


def save_game_entity(game_id: str, game_doc: dict):
    if USE_LOCAL_STORE:
        LOCAL_GAMES[game_id] = game_doc
        return

    # Uncomment for GCP deployment
    entity = datastore.Entity(key=ds.key(GAMES_KIND, game_id))
    entity.update(game_doc)
    ds.put(entity)


def save_move_entity(move_id: str, move_doc: dict):
    if USE_LOCAL_STORE:
        LOCAL_MOVES[move_id] = move_doc
        return

    # Uncomment for GCP deployment
    entity = datastore.Entity(key=ds.key(MOVES_KIND, move_id))
    entity.update(move_doc)
    ds.put(entity)


def player_from_token(game: Dict[str, Any], token: str) -> str:
    players = game.get("players", {})
    if players.get("P1", {}).get("token") == token:
        return "P1"
    if players.get("P2", {}).get("token") == token:
        return "P2"
    raise HTTPException(403, "Invalid player token")


# -----------------------------------
# Chess logic
# -----------------------------------
def apply_human_move(state: Dict[str, Any], move: Dict[str, Any], player: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if state.get("status") != "ACTIVE":
        raise HTTPException(400, "Game is not active")

    fen = state.get("fen")
    if not fen:
        raise HTTPException(500, "Missing FEN in game state")

    board = chess.Board(fen)

    expected_player = "P1" if board.turn == chess.WHITE else "P2"
    if player != expected_player:
        raise HTTPException(400, f"Not {player}'s turn")

    chess_move = move_dict_to_chess_move(move, board)
    if chess_move not in board.legal_moves:
        raise HTTPException(400, f"Illegal move: {chess_move.uci()}")

    san = board.san(chess_move)
    uci = chess_move.uci()
    board.push(chess_move)

    new_fen = board.fen()
    new_state = dict(state)
    new_state["fen"] = new_fen
    new_state["board"] = fen_to_flat_board(new_fen)
    new_state["turn"] = "P1" if board.turn == chess.WHITE else "P2"
    new_state["status"] = game_status_from_board(board)
    new_state["message"] = game_message_from_board(board, player, san)
    new_state["lastMoveSan"] = san
    new_state["lastMoveUci"] = uci

    move_payload = {
        "from": move["from"],
        "to": move["to"],
        "san": san,
        "uci": uci,
    }

    return new_state, move_payload


def get_ai_move_from_model(current_fen: str, difficulty: str = "medium") -> str:
    """
    Returns a UCI move like 'e2e4' or 'g1f3'.
    """
    try:
        model = get_model()
        return predict_best_move(current_fen, model, MODEL_DEVICE)
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {str(e)}")


def apply_ai_move(state: Dict[str, Any], difficulty: str) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    if state.get("status") != "ACTIVE":
        raise HTTPException(400, "Game is not active")

    fen = state.get("fen")
    if not fen:
        raise HTTPException(500, "Missing FEN in game state")

    board = chess.Board(fen)
    ai_player = "P1" if board.turn == chess.WHITE else "P2"

    uci = get_ai_move_from_model(fen, difficulty)

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        raise HTTPException(500, f"Model returned invalid UCI: {uci}")

    if move not in board.legal_moves:
        raise HTTPException(500, f"Model returned illegal move: {uci}")

    san = board.san(move)

    from_sq = move.from_square
    to_sq = move.to_square
    from_payload = {
        "r": 7 - chess.square_rank(from_sq),
        "c": chess.square_file(from_sq),
    }
    to_payload = {
        "r": 7 - chess.square_rank(to_sq),
        "c": chess.square_file(to_sq),
    }

    board.push(move)

    new_fen = board.fen()
    new_state = dict(state)
    new_state["fen"] = new_fen
    new_state["board"] = fen_to_flat_board(new_fen)
    new_state["turn"] = "P1" if board.turn == chess.WHITE else "P2"
    new_state["status"] = game_status_from_board(board)
    new_state["message"] = game_message_from_board(board, ai_player, san)
    new_state["lastMoveSan"] = san
    new_state["lastMoveUci"] = uci

    move_payload = {
        "from": from_payload,
        "to": to_payload,
        "san": san,
        "uci": uci,
        "difficulty": difficulty,
    }

    return new_state, move_payload, ai_player


def auto_play_ai_if_needed(game_id: str, game: dict, state: dict) -> dict:
    """
    In PVAI mode, if it is P2's turn after a move, immediately make the AI move.
    """
    if game.get("mode") != "PVAI":
        return state

    if state.get("status") != "ACTIVE":
        return state

    if state.get("turn") != "P2":
        return state

    ai_state, ai_move_payload, ai_player = apply_ai_move(state, "medium")

    ai_move_id = new_public_id("move")
    ai_move_doc = {
        "moveId": ai_move_id,
        "gameId": game_id,
        "player": ai_player,
        "move": ai_move_payload,
        "clientTs": None,
        "serverTs": now_iso(),
        "type": "AI",
    }
    save_move_entity(ai_move_id, ai_move_doc)

    return ai_state


# -----------------------------------
# Routes
# -----------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "storage": "local" if USE_LOCAL_STORE else "datastore",
    }


@app.post("/games", response_model=CreateGameResponse)
def create_game(req: CreateGameRequest):
    game_id = new_public_id("game")
    join_code = secrets.token_hex(3) if req.mode == "PVP" else None
    p1_token = secrets.token_urlsafe(16)
    state = initial_state()

    doc = {
        "gameId": game_id,
        "joinCode": join_code,
        "mode": req.mode,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "players": {
            "P1": {"token": p1_token},
            "P2": {"token": None},
        },
        "state": state,
    }

    save_game_entity(game_id, doc)

    return CreateGameResponse(
        gameId=game_id,
        joinCode=join_code,
        player="P1",
        playerToken=p1_token,
        state=state,
    )


@app.post("/join", response_model=JoinResponse)
def join_game(req: JoinRequest):
    game = find_game_by_join_code(req.joinCode)

    if game.get("mode") == "PVAI":
        raise HTTPException(400, "This game is configured for player vs AI")

    players = game.get("players", {})
    if players.get("P2", {}).get("token"):
        raise HTTPException(409, "Game already has two players")

    p2_token = secrets.token_urlsafe(16)
    players["P2"] = {"token": p2_token}

    state = game.get("state", {})
    state["message"] = "P2 joined the game"

    game["players"] = players
    game["state"] = state
    game["updatedAt"] = now_iso()
    save_game_entity(game["gameId"], game)

    return JoinResponse(
        gameId=game["gameId"],
        player="P2",
        playerToken=p2_token,
        state=state,
    )


@app.get("/games/{game_id}")
def get_game(game_id: str):
    game = get_game_entity(game_id)

    return {
        "gameId": game["gameId"],
        "createdAt": game.get("createdAt"),
        "updatedAt": game.get("updatedAt"),
        "mode": game.get("mode", "PVP"),
        "state": game.get("state"),
    }


@app.post("/games/{game_id}/move", response_model=MoveResponse)
def make_move(game_id: str, req: MoveRequest):
    game = get_game_entity(game_id)

    player = player_from_token(game, req.playerToken)
    state = game.get("state", {})

    if game.get("mode") == "PVAI" and player != "P1":
        raise HTTPException(403, "Only P1 can make human moves in player vs AI mode")

    new_state, move_payload = apply_human_move(state, req.move, player)

    move_id = new_public_id("move")
    move_doc = {
        "moveId": move_id,
        "gameId": game_id,
        "player": player,
        "move": move_payload,
        "clientTs": req.clientTs,
        "serverTs": now_iso(),
        "type": "HUMAN",
    }
    save_move_entity(move_id, move_doc)

    # new_state = auto_play_ai_if_needed(game_id, game, new_state)

    game["state"] = new_state
    game["updatedAt"] = now_iso()
    save_game_entity(game_id, game)

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)


@app.post("/games/{game_id}/ai-move", response_model=MoveResponse)
def ai_move(game_id: str, req: AiMoveRequest):
    game = get_game_entity(game_id)
    state = game.get("state", {})

    if req.playerToken:
        player = player_from_token(game, req.playerToken)
        expected_player = state.get("turn")
        if player != expected_player:
            raise HTTPException(403, "Only the player whose turn it is can request the AI move")

    new_state, move_payload, ai_player = apply_ai_move(state, req.difficulty or "medium")

    move_id = new_public_id("move")
    move_doc = {
        "moveId": move_id,
        "gameId": game_id,
        "player": ai_player,
        "move": move_payload,
        "clientTs": None,
        "serverTs": now_iso(),
        "type": "AI",
    }
    save_move_entity(move_id, move_doc)

    # IMPORTANT FIX:
    # In PVAI mode, if P1 asked AI to move on P1's turn, now it becomes P2's turn.
    # Auto-play the bot response too.
    # new_state = auto_play_ai_if_needed(game_id, game, new_state)

    game["state"] = new_state
    game["updatedAt"] = now_iso()
    save_game_entity(game_id, game)

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)