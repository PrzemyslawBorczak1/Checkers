#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdint>
#include <cstring>

#include "common.h"

#define THREADS 128
#define BLOCKS 512


// move_packed: next_enemy | dir | type | from | with | to
# define MOVE_TYPE_NORMAL 0u
# define MOVE_TYPE_PAWN_CAPTURE 1u
# define MOVE_TYPE_KING_FINAL_CAPTURE 3u
# define MOVE_TYPE_KING_BRANCHING_CAPTURE 2u


#define NO_PROGRES_LIMIT 30 // po 15 ruchow z dwoch stron bez progresu gra jest przerywana (przegrywa ten ktory wykon aruch bez progresu)
#define MIN_MOVES 200 // po 100 ruchach z kazdej strony mozliwe jest przerwanie gry przy znaczenej przewadze (3 razy wiecej materialu)


uint32_t runMCTS(Board* dev_board, Color color, uint32_t seed);
cudaError_t mctsSetSymbols(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
