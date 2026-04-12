"""
agent.py — self-contained chess agent: tokenizer + model + move selection.
"""
from __future__ import annotations
import dataclasses
import os

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def tokenize_fen(fen: str) -> list[int]:

    # ---------------------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------------------
    _CHARACTERS = [
        '0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h',
        'p','n','r','k','q',
        'P','B','N','R','Q','K',
        'w','.'
    ]
    _CHAR_TO_IDX  = {c: i for i, c in enumerate(_CHARACTERS)}
    NUM_FEN_CHARS = len(_CHARACTERS)   # 31
    FEN_LENGTH    = 77

    parts = fen.strip().split()
    board_field, active, castling, ep, halfmove, fullmove = parts[:6]

    board = ""
    for ch in board_field:
        if ch == '/':
            continue
        board += '.' * int(ch) if ch.isdigit() else ch

    castling_str = "".join(c if c in castling else '.' for c in "KQkq")
    ep_str       = '..' if ep == '-' else ep.ljust(2, '.')[:2]

    full = board + active + castling_str + ep_str + halfmove.zfill(3)[:3] + fullmove.zfill(3)[:3]
    return [_CHAR_TO_IDX[c] for c in full]


# ---------------------------------------------------------------------------
# Move vocabulary
# ---------------------------------------------------------------------------
def _build_vocab() -> dict[str, int]:
    moves = set()
    def sq(fi, ri): return chr(ord('a') + fi) + str(ri + 1)
    for sfi in range(8):
        for sri in range(8):
            src = sq(sfi, sri)
            for dfi in range(8):
                if dfi != sfi: moves.add(src + sq(dfi, sri))
            for dri in range(8):
                if dri != sri: moves.add(src + sq(sfi, dri))
            for d in range(1, 8):
                for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nf, nr = sfi + d*dx, sri + d*dy
                    if 0 <= nf < 8 and 0 <= nr < 8: moves.add(src + sq(nf, nr))
            for dx, dy in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
                nf, nr = sfi + dx, sri + dy
                if 0 <= nf < 8 and 0 <= nr < 8: moves.add(src + sq(nf, nr))
    promo = set()
    for m in moves:
        if (int(m[1]) == 7 and int(m[3]) == 8) or (int(m[1]) == 2 and int(m[3]) == 1):
            if abs(ord(m[0]) - ord(m[2])) <= 1:
                for p in 'qrbn': promo.add(m + p)
    moves |= promo
    moves = sorted(moves)
    assert len(moves) == 1968
    return {m: i for i, m in enumerate(moves)}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    K: int = 128; D: int = 256; layers: int = 8; heads: int = 8


class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H; self.Dh = D // H
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)

    def forward(self, x):
        B, T, D = x.shape; H, Dh = self.H, self.Dh
        q = self.q(x).view(B,T,H,Dh).transpose(1,2)
        k = self.k(x).view(B,T,H,Dh).transpose(1,2)
        v = self.v(x).view(B,T,H,Dh).transpose(1,2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1,2).reshape(B,T,D))


class SwiGLU(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.g = nn.Linear(D, 4*D, bias=False)
        self.u = nn.Linear(D, 4*D, bias=False)
        self.d = nn.Linear(4*D, D, bias=False)

    def forward(self, x): return self.d(F.silu(self.g(x)) * self.u(x))


class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.attn = MultiHeadAttention(D, H)
        self.ln2 = nn.LayerNorm(D); self.mlp  = SwiGLU(D)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ActionValueModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        MOVE_TO_ACTION = _build_vocab()
        NUM_ACTIONS    = len(MOVE_TO_ACTION)
        SEQ_LENGTH     = 78

        self.cfg        = cfg
        self.fen_embed  = nn.Embedding(31, cfg.D) # len(_CHARACTERS)   # 31
        self.move_embed = nn.Embedding(NUM_ACTIONS,   cfg.D)
        self.pos        = nn.Embedding(SEQ_LENGTH,    cfg.D)
        self.blocks     = nn.ModuleList([Block(cfg.D, cfg.heads) for _ in range(cfg.layers)])
        self.ln         = nn.LayerNorm(cfg.D)
        self.head       = nn.Linear(cfg.D, cfg.K)
        self.register_buffer("pos_ids", torch.arange(SEQ_LENGTH))

    def forward(self, fen, move):
        x = self.fen_embed(fen)
        m = self.move_embed(move).unsqueeze(1)
        x = torch.cat([x, m], dim=1) + self.pos(self.pos_ids)
        for b in self.blocks: x = b(x)
        return F.log_softmax(self.head(self.ln(x)[:, -1]), dim=-1)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(checkpoint: str, device: torch.device) -> ActionValueModel:
    model = ActionValueModel(Config()).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Best move
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_best_move(fen: str, model: ActionValueModel, device: torch.device) -> str:
    """
    Generate all legal moves for the position, score each with the model,
    and return the UCI move with the highest expected win probability.
    """

    MOVE_TO_ACTION = _build_vocab()
    NUM_ACTIONS    = len(MOVE_TO_ACTION)
    SEQ_LENGTH     = 78

    board  = chess.Board(fen)
    legals = [m.uci() for m in board.legal_moves if m.uci() in MOVE_TO_ACTION]
    if not legals:
        raise ValueError(f"No legal moves in vocabulary for: {fen}")

    fen_tokens = tokenize_fen(fen)
    fen_t  = torch.tensor(fen_tokens, dtype=torch.long, device=device).unsqueeze(0).expand(len(legals), -1)
    move_t = torch.tensor([MOVE_TO_ACTION[m] for m in legals], dtype=torch.long, device=device)

    log_probs = model(fen_t, move_t)                          # (N, K)

    K     = model.cfg.K
    mids  = torch.linspace(0.5/K, 1 - 0.5/K, K, device=device)
    wins  = (log_probs.exp() * mids).sum(dim=-1)              # expected win prob per move

    return legals[wins.argmax().item()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = os.path.join(os.path.dirname(__file__), "latest.pt")
    model  = load_model(ckpt, device)
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)  device={device}\n")

    tests = [
        # (description, FEN)
        (
            "Starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "After 1.e4 (black to move)",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ),
        (
            "Sicilian after 1.e4 c5 2.Nf3 (white to move)",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 2",
        ),
        (
            "Mate in 1 (white): Qxf7#",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        ),
        (
            "Endgame: K+R vs K (white to move)",
            "8/8/8/8/3k4/8/8/3KR3 w - - 0 1",
        ),
    ]

    for desc, fen in tests:
        move  = predict_best_move(fen, model, device)
        board = chess.Board(fen)
        print(f"{desc}")
        print(f"  FEN  : {fen}")
        print(f"  Move : {move}")
        print("\nBefore:")
        print(board)
        board.push(chess.Move.from_uci(move))
        print(f"\nAfter {move}:")
        print(board)
        print()
