#pragma once
#include <vector>    
#include "common.h"
#include "player.h"



class HumanPlayer : public Player {
private:
   
   
    bool parseMove(char* move_str, vector<int>& steps, char delim);

    vector<int> normalizeMove(vector<int>& steps);

    bool moveEqualsNormalized(vector<int>& a, vector<int>& b);

    PossibleMove* findMoveNormalized(vector<PossibleMove>& pm, vector<int>& m);

    void printMoveVector(vector<PossibleMove> pm);

public:
    HumanPlayer(Color c) : Player(c) {}

    void MakeMove(Board& board, char* ret, int moves_without_progress);

};



