#pragma once
#include <stdio.h>
#include "common.h"
#include "player.h"
#include "moves_getter.h" 
#include "GPU.h"


class MCTSPlayer : public Player {
private:
	int time_limit_sec;
	GPU gpu;

public:
    MCTSPlayer(Color c, int time_limit_sec) : Player(c), time_limit_sec(time_limit_sec), gpu(GPU::getInstance(Neighbours, Captures, Rays)){
    }

    int simulate(Board board) {
		gpu.simulate(board, color);
        // Placeholder for simulation logic
        return 0;
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
			int wins = simulate(pm.resulting_board);
			printf("wins: %d\n", wins);
		}



        return "\0";
    }
};
