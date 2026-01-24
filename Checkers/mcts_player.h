#pragma once
#include <stdio.h>
#include "common.h"
#include "player.h"


class MCTSPlayer : public Player {
private:
	int time_limit_sec;

public:
    MCTSPlayer(Color c, int time_limit_sec) : Player(c), time_limit_sec(time_limit_sec) {}

    char* MakeMove(Board& board) override {
        (void)board;
        // TODO: tu MCTS
        // Na razie: stub
        return "\0";
    }
};
