#pragma once
#include <stdint.h>

#include "cuda_runtime.h"

// czarne na gorze
// uklad
// notacja
//   a b c d e f g h
//
//      7   3
//      6   2
// ..   5   1
// 8    4   0

struct Board {
    uint32_t white_pawns;
    uint32_t white_kings;
    uint32_t black_pawns;
    uint32_t black_kings;
};

enum class Color {
    WHITE,
    BLACK,
	UNDEFINED
};


#define NO_PROGRESS_LIMIT 30

__host__ __device__ void writeBoardToBuff(char buffer[72], uint32_t board, char c);

 __host__ __device__ void printBoard(uint32_t black_pawns,
    uint32_t white_pawns,
    uint32_t black_kings,
    uint32_t white_kings);

__host__ __device__ void printBoard(Board board);


Board startBoard();