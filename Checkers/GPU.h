#pragma once
#include "common.h"

class GPU {
public:
	Board* dev_board;


    static GPU& getInstance(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
	// zwraca ilosc wygrancyh przez bialego, -1 na blad
    int simulate(Board board, Color color, uint32_t seed, int moves_without_progress);

private:

    GPU(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
    static GPU* instance_;
    void gpuPrelude( int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
};