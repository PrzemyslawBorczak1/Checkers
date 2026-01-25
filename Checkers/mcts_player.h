#pragma once
#include <stdio.h>
#include "common.h"
#include "player.h"
#include "moves_getter.h" 
#include "GPU.h"
#include <chrono>
#include <iostream>


class MCTSPlayer : public Player {
private:
	int time_limit_sec;
	GPU gpu;

public:
    MCTSPlayer(Color c, int time_limit_sec) : Player(c), time_limit_sec(time_limit_sec), gpu(GPU::getInstance(Neighbours, Captures, Rays)){
    }

    int simulate(Board board, Color next_color) {
		uint32_t ret = gpu.simulate(board, next_color);
        // Placeholder for simulation logic
        return ret;
	}

    char* MakeMove(Board& board) override {
		printf("MCTS Player making move:\n");
		vector<PossibleMove> possible_moves = getAllMoves(board, color);
        for(auto pm : possible_moves) {
            printf("[ ");
            for(int pos : pm.move) {
                printf("%d ", pos);
            }

            printf("]\n");

            auto t0 = std::chrono::high_resolution_clock::now();

            int wins = simulate(pm.resulting_board, Color::BLACK);

            auto t1 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "simulate() took " << us << " ms\n";
			printf("wins: %d\n", wins);
		}



        return "\0";
    }
};
