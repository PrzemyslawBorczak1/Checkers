# Checkers (CUDA + MCTS)

This project implements a checkers game where moves are selected using the MCTS (Monte Carlo Tree Search) algorithm. Rollout simulations are executed in parallel on the GPU (CUDA) to speed up move evaluation.

---

## Usage

checkers file time game_type color

file      game save file
time      per-move time limit in seconds, format T1:T2 (T1 for White, T2 for Black)
game_type 0 = human-human, 1 = human-computer, 2 = computer-computer
color     0 = white, 1 = black (in mode 1 this is the human player's color; in other modes it does not matter)

Examples:
checkers game.txt 1:2 0 0
checkers game.txt 5:1 1 1
checkers game.txt 10:1 2 0

---

To play you need to pass moves in the following format:
- For normal moves, separate the starting square and the finishing square with "-" (e.g., d2-e3).
- For captures, separate successive squares with ":" (e.g., d2:f4:d6).

## Checkers rules
- The game is played on an 8x8 board. Pieces are placed on black squares only.
- Both sides start with 12 pawns.
- Captures are performed by jumping over an enemy piece and landing on an empty square.
- Backward captures are allowed and all captures are forced. If a multi-capture is available, the piece must continue capturing until no further captures are possible.
- A pawn becomes a king only after it finishes its move on the last row.
- Kings can move any number of squares.
- If a side has no legal moves on its turn, it loses.
- If there are 30 moves with no progress, the game is a draw.

## Algorithm overview (MCTS)

The program selects moves using MCTS. One iteration consists of:
- selection,
- expansion,
- simulation (rollout),
- backpropagation (updating statistics in the tree).

### UCT

During selection, the child with the highest UCT value is chosen:

UCT(i) = (w_i / n_i) + c * sqrt( ln(N) / n_i )

where:
- w_i – number of wins (or reward) for the node,
- n_i – number of visits of the node,
- N – number of visits of the parent,
- c – exploration constant.

---

## GPU rollout

Rollout simulations are executed in parallel on the GPU.

Number of simulations per batch:
BLOCKS * THREADS

- THREADS – number of threads per block,
- BLOCKS – number of blocks, typically a few hundred blocks with 128 threads each (depending on the GPU).

Each thread runs one simulation on its own copy of the board. The simplified move logic in rollout:
- first, pawn captures are checked,
- if a capture exists, a capture variant is chosen randomly,
- then kings are handled similarly (captures and moves),
- if no capture exists, normal pawn moves are performed.

### Early termination conditions

A simulation can end early:
- after 200 half-moves (100 moves per side) if one side has a material advantage of at least 3:1 (a king counts as 2 pawns),
- after 30 moves without progress (no captures / no pawn move), the side with more material wins,
- if material is equal, the side that made the 31st move loses.

---

## Data sent to the GPU

The GPU receives the board and metadata (e.g., side to move and the number of moves without progress).

Board representation (bitboards):

struct Boards {
    uint32_t* white_pawns;
    uint32_t* black_pawns;
    uint32_t* white_kings;
    uint32_t* black_kings;
};

---

## Result reduction

Each thread writes its rollout result (0/1) to shared memory. Results are then reduced (summed) within each block. A separate kernel performs the reduction across blocks.

---

## Performance

This implementation achieved millions of simulations per second on an RTX 3050 and hundreds of millions of simulations per second on an RTX 5090.

The checkers rules implemented here are based on the rules I used when playing as a child. In practice, this ruleset tends to produce many draws, but I personally could not beat this engine.
