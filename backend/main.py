from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import datastore
from datetime import datetime, timezone
import secrets
from typing import Any, Dict, Optional

app = FastAPI(title="Shatigo Game API (Datastore)")

# CORS (okay for local dev; lock down later)
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

ds = datastore.Client()

GAMES_KIND = "Game"
MOVES_KIND = "Move"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_public_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(10)}"


def initial_state() -> Dict[str, Any]:
    # Datastore: no nested arrays. Use flat 64-length list.
    return {
        "board": ["r","n","b","q","k","b","n","r",
                "p","p","p","p","p","p","p","p",
                ".",".",".",".",".",".",".",".",
                ".",".",".",".",".",".",".",".",
                ".",".",".",".",".",".",".",".",
                ".",".",".",".",".",".",".",".",
                "P","P","P","P","P","P","P","P",
                "R","N","B","Q","K","B","N","R"],#["." for _ in range(64)],  # 8x8 flattened row-major
        "turn": "P1",
        "status": "ACTIVE",
        "message": "Game created",
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


def get_game_entity(game_id: str):
    entity = ds.get(ds.key(GAMES_KIND, game_id))
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


def validate_and_apply_move(state: Dict[str, Any], move: Dict[str, Any], player: str) -> Dict[str, Any]:
    """
    TODO: implement real Shatigo rules here.
    For now: checks turn + swaps turn + stores message.
    """
    if state.get("status") != "ACTIVE":
        raise HTTPException(400, "Game is not active")

    if state.get("turn") != player:
        raise HTTPException(400, f"Not {player}'s turn")

    new_state = dict(state)
    new_state["message"] = f"{player} moved: {move}"
    new_state["turn"] = "P2" if player == "P1" else "P1"
    return new_state


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/games", response_model=CreateGameResponse)
def create_game():
    game_id = new_public_id("game")
    join_code = secrets.token_hex(3)  # short join code
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

    entity = datastore.Entity(key=ds.key(GAMES_KIND, game_id))
    entity.update(doc)
    ds.put(entity)

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
    ds.put(entity)

    return JoinResponse(gameId=game_id, player="P2", playerToken=p2_token, state=state)


@app.get("/games/{game_id}")
def get_game(game_id: str):
    entity = get_game_entity(game_id)
    game = dict(entity)
    # Do not leak tokens
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

    new_state = validate_and_apply_move(state, req.move, player)

    move_id = new_public_id("move")
    move_entity = datastore.Entity(key=ds.key(MOVES_KIND, move_id))
    move_entity.update({
        "moveId": move_id,
        "gameId": game_id,
        "player": player,
        "move": req.move,
        "clientTs": req.clientTs,
        "serverTs": now_iso(),
        "type": "HUMAN",
    })

    # Save move + update game
    ds.put_multi([move_entity])
    entity.update({"state": new_state, "updatedAt": now_iso()})
    ds.put(entity)

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)


@app.post("/games/{game_id}/ai-move", response_model=MoveResponse)
def ai_move(game_id: str, req: AiMoveRequest):
    """
    Stub AI move. Later: call your ML model endpoint to get a real move.
    """
    entity = get_game_entity(game_id)
    game = dict(entity)

    state = game.get("state", {})
    player = state.get("turn", "P2")  # AI plays current turn for now

    dummy_move = {"type": "AI_DUMMY", "difficulty": req.difficulty}
    new_state = validate_and_apply_move(state, dummy_move, player)

    move_id = new_public_id("move")
    move_entity = datastore.Entity(key=ds.key(MOVES_KIND, move_id))
    move_entity.update({
        "moveId": move_id,
        "gameId": game_id,
        "player": player,
        "move": dummy_move,
        "clientTs": None,
        "serverTs": now_iso(),
        "type": "AI",
    })

    ds.put_multi([move_entity])
    entity.update({"state": new_state, "updatedAt": now_iso()})
    ds.put(entity)

    return MoveResponse(gameId=game_id, state=new_state, moveId=move_id)
