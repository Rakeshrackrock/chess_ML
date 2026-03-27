from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from google.cloud import datastore
from datetime import datetime, timezone
import secrets
import random
import chess
from typing import Any, Dict, Optional

app = FastAPI(title="Chess Game API (Datastore)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://shatigoai.web.app",
        "https://shatigoai.firebaseapp.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ds = datastore.Client()

# ---- LOCAL DEV STORAGE ----
LOCAL_GAMES = {}
LOCAL_MOVES = {}

USE_LOCAL_STORE = True

GAMES_KIND = "Game"
MOVES_KIND = "Move"


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
    Auto-promotes pawns to queen for now.
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
        winner = last_actor
        return f"{last_san} — checkmate. {winner} wins."
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


class CreateGameResponse(BaseModel):
    gameId: str
    joinCode: str
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


# def get_game_entity(game_id: str):
#     entity = ds.get(ds.key(GAMES_KIND, game_id))
#     if not entity:
#         raise HTTPException(404, "Game not found")
#     return entity

def get_game_entity(game_id: str):
    entity = LOCAL_GAMES.get(game_id)
    if not entity:
        raise HTTPException(404, "Game not found")
    return entity


def player_from_token(game: Dict[str, Any], token: str) -> str:
    players = game.get("players", {})
    if players.get("P1", {}).get("token") == token:
        return "P1"
    if players.get("P2", {}).get("token") == token:
        return "P2"
    raise HTTPException(403, "Invalid player token")


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
    Replace this with your ML team's real inference function.

    Expected contract:
      input  -> current_fen (str)
      output -> SAN move (str), e.g. 'e4', 'Nf3', 'Qxe5+', 'O-O'
    """
    board = chess.Board(current_fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise HTTPException(400, "No legal moves available for AI")

    # Temporary fallback so integration works now.
    move = random.choice(legal_moves)
    return board.san(move)


def apply_ai_move(state: Dict[str, Any], difficulty: str) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    if state.get("status") != "ACTIVE":
        raise HTTPException(400, "Game is not active")

    fen = state.get("fen")
    if not fen:
        raise HTTPException(500, "Missing FEN in game state")

    board = chess.Board(fen)
    ai_player = "P1" if board.turn == chess.WHITE else "P2"

    san = get_ai_move_from_model(fen, difficulty)

    try:
        move = board.parse_san(san)
    except ValueError:
        raise HTTPException(500, f"Model returned invalid SAN: {san}")

    if move not in board.legal_moves:
        raise HTTPException(500, f"Model returned illegal move: {san}")

    uci = move.uci()

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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/games", response_model=CreateGameResponse)
def create_game():
    game_id = new_public_id("game")
    join_code = secrets.token_hex(3)
    p1_token = secrets.token_urlsafe(16)
    state = initial_state()

    doc = {
        "gameId": game_id,
        "joinCode": join_code,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "players": {
            "P1": {"token": p1_token},
            "P2": {"token": None},
        },
        "state": state,
    }

    # entity = datastore.Entity(key=ds.key(GAMES_KIND, game_id))
    # entity.update(doc)
    # ds.put(entity)

    LOCAL_GAMES[game_id] = doc

    return CreateGameResponse(
        gameId=game_id,
        joinCode=join_code,
        player="P1",
        playerToken=p1_token,
        state=state,
    )


@app.post("/games/{game_id}/join", response_model=JoinResponse)
def join_game(game_id: str, req: JoinRequest):
    entity = get_game_entity(game_id)
    game = dict(entity)

    if game.get("joinCode") != req.joinCode:
        raise HTTPException(403, "Invalid join code")

    players = game.get("players", {})
    if players.get("P2", {}).get("token"):
        raise HTTPException(409, "Game already has two players")

    p2_token = secrets.token_urlsafe(16)
    players["P2"] = {"token": p2_token}

    state = game.get("state", {})
    state["message"] = "P2 joined the game"

    entity.update({
        "players": players,
        "state": state,
        "updatedAt": now_iso(),
    })
    # ds.put(entity)
    LOCAL_GAMES[game_id] = entity

    return JoinResponse(
        gameId=game_id,
        player="P2",
        playerToken=p2_token,
        state=state,
    )


@app.get("/games/{game_id}")
def get_game(game_id: str):
    entity = get_game_entity(game_id)
    game = dict(entity)

    return {
        "gameId": game["gameId"],
        "createdAt": game.get("createdAt"),
        "updatedAt": game.get("updatedAt"),
        "state": game.get("state"),
    }


@app.post("/games/{game_id}/move", response_model=MoveResponse)
def make_move(game_id: str, req: MoveRequest):
    entity = get_game_entity(game_id)
    game = dict(entity)

    player = player_from_token(game, req.playerToken)
    state = game.get("state", {})

    new_state, move_payload = apply_human_move(state, req.move, player)

    move_id = new_public_id("move")
    # move_entity = datastore.Entity(key=ds.key(MOVES_KIND, move_id))
    # move_entity.update({
    #     "moveId": move_id,
    #     "gameId": game_id,
    #     "player": player,
    #     "move": move_payload,
    #     "clientTs": req.clientTs,
    #     "serverTs": now_iso(),
    #     "type": "HUMAN",
    # })

    LOCAL_MOVES[move_id] = {
        "moveId": move_id,
        "gameId": game_id,
        "player": player,
        "move": move_payload,
        "clientTs": req.clientTs,
        "serverTs": now_iso(),
        "type": "HUMAN",
}

    # ds.put_multi([move_entity])

    entity.update({
        "state": new_state,
        "updatedAt": now_iso(),
    })
    # ds.put(entity)
    LOCAL_GAMES[game_id] = entity

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)


@app.post("/games/{game_id}/ai-move", response_model=MoveResponse)
def ai_move(game_id: str, req: AiMoveRequest):
    entity = get_game_entity(game_id)
    game = dict(entity)
    state = game.get("state", {})

    new_state, move_payload, ai_player = apply_ai_move(state, req.difficulty or "medium")

    move_id = new_public_id("move")
    # move_entity = datastore.Entity(key=ds.key(MOVES_KIND, move_id))
    # move_entity.update({
    #     "moveId": move_id,
    #     "gameId": game_id,
    #     "player": ai_player,
    #     "move": move_payload,
    #     "clientTs": None,
    #     "serverTs": now_iso(),
    #     "type": "AI",
    # })
    LOCAL_MOVES[move_id] = {
        "moveId": move_id,
        "gameId": game_id,
        "player": player,
        "move": move_payload,
        "clientTs": req.clientTs,
        "serverTs": now_iso(),
        "type": "HUMAN",
}
    # ds.put_multi([move_entity])

    entity.update({
        "state": new_state,
        "updatedAt": now_iso(),
    })
    # ds.put(entity)
    LOCAL_GAMES[game_id] = entity

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)