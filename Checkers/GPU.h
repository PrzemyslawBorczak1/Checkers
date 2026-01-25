#pragma once
#include "common.h"

#define THREADS_PER_BLOCK 1
#define BLOCKS 1

class GPU {
public:
	Board* dev_board;
	uint16_t* dev_ret;


    static GPU& getInstance(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
	// zwraca ilosc wygrancyh przez bialego, -1 na blad
    int simulate(Board board, Color color);

private:

    GPU(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
    static GPU* instance_;
    void gpuPrelude( int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
};