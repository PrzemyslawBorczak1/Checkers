#pragma once
#include <stdio.h>
#include <string.h>
#include "common.h"



class Player {
protected:
    Color color;

public:
    Player(Color c) : color(c) {}
    virtual ~Player() {}

    virtual char* MakeMove(Board& board, int time_sec) = 0;
};



class HumanPlayer : public Player {
private:
    char buf[128];

public:
    HumanPlayer(Color c) : Player(c) { buf[0] = '\0'; }

    char* MakeMove(Board& board, int time_sec) override {
        (void)board; (void)time_sec;
        printf("Podaj ruch: ");
        fflush(stdout);

        if (!fgets(buf, (int)sizeof(buf), stdin)) {
            buf[0] = '\0';
            return buf;
        }
        size_t n = strlen(buf);
        if (n > 0 && buf[n - 1] == '\n') buf[n - 1] = '\0';
        return buf;
    }
};





class AiPlayer : public Player {
private:
    char buf[128];

public:
    AiPlayer(Color c) : Player(c) { buf[0] = '\0'; }

    char* MakeMove(Board& board, int time_sec) override {
        (void)board; (void)time_sec;
        // TODO: tu MCTS
        // Na razie: stub
        strcpy(buf, "d2-e3");
        return buf;
    }
};
