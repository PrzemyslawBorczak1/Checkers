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
    BLACK
};


__host__ __device__ void writeBoardToBuff(char buffer[72], uint32_t board, char c);

 __host__ __device__ void printBoard(uint32_t black_pawns,
    uint32_t white_pawns,
    uint32_t black_kings,
    uint32_t white_kings);

__host__ __device__ void printBoard(Board board);


__host__ __device__ void print_int(uint32_t n);

__host__ __device__ void print_bin(uint32_t n);

// ---- gotowe pozycje testowe ----
Board startBoard();
Board captureBoard();
Board endBoard();
Board backCapture();
Board firstRow();
Board lastRow1();
Board lastRow2();

Board cycleBoard1();
Board cycleBoard2();
Board cycleBoard3();

Board kingMainLine1();
Board kingMainLine2();

Board kingLine1();
Board kingLine2();
Board kingLine3();
Board kingLine4();
Board kingLine5();

Board kingAllDir();
Board kingCycle();
Board kingWierd();

Board kingBlocking1();
Board kingBlocking2();
Board kingMultiple();

Board kingPawnsCapture();
Board kingPawnsNoCapture();
Board kingPawnsBlack();

Board promotion();

Board captureNoPromotion();
Board capturePromotion();

Board kingArena();
Board whiteLeading();
Board blackLeading();