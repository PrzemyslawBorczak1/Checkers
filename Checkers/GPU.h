#pragma once
#include "common.h"

// singleton obslugujacy GPU
class GPU {
public:
	Board* dev_board;
	uint32_t* dev_ret;


    static GPU& getInstance(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
	// zwraca ilosc wygrancyh przez bialego, -1 na blad
    int simulate(Board board, Color color, uint32_t seed, int moves_without_progress);

private:
	// wykonywany raz jako ze jest to singleton
	// funkcja inicjuje tablice na GPU oraz alokuje na niej pamiec
    GPU(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
    static GPU* instance_;
    void gpuPrelude( int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]);
};