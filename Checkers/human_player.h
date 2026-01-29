#pragma once
#include <vector>    
#include "common.h"
#include "player.h"



class HumanPlayer : public Player {
private:
   
    // odczyt inputu z termiannla
    bool parseMove(char* move_str, vector<int>& steps, char delim);

    // zapis w formie bez powtarzenia ruchow na tej samej prostej
    vector<int> normalizeMove(vector<int>& steps);

	// sprawdzenie czy dwa ruchy sa sobie rowne po normalizacji
    bool moveEqualsNormalized(vector<int>& a, vector<int>& b);

	// sprawdzenie czy ruch podnay przez gracza jest w vecotrze legalnych ruchow
    PossibleMove* findMoveNormalized(vector<PossibleMove>& pm, vector<int>& m);

    void printMoveVector(vector<PossibleMove> pm);

public:
    HumanPlayer(Color c) : Player(c) {}

	// glowna funkcja klasy odpowidziealna za wykonanie tury gracza
    void MakeMove(Board& board, char* ret, int moves_without_progress);

};



