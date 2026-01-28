#pragma once
#include "player.h"
#include "GPU.h"
#include <vector>

#define EXPLORE_CONSTANT 1.41


struct MCTSTreeNode {
    PossibleMove possible_move;
    Color side_to_move;

    uint64_t visits;
    uint64_t wins;

    MCTSTreeNode* parent = nullptr;

	bool is_expanded = false;
    vector<MCTSTreeNode*> children;
    MCTSTreeNode(PossibleMove pm, Color c) : possible_move(pm), side_to_move(c), visits(0), wins(0) {}
};


class MCTSPlayer : public Player {
private:
    int time_limit_sec;
    GPU gpu;
    MCTSTreeNode* root;

public:
    MCTSPlayer(Color c, int time_limit_sec) : Player(c), time_limit_sec(time_limit_sec), gpu(GPU::getInstance(Neighbours, Captures, Rays)) {
    }


    void printCollectedStats(const MCTSTreeNode* root, uint64_t counter);

    // wykonuje symulacji MTCS na GPU
    int simulate(const Board board, Color next_color, uint32_t seed, int moves_without_progress);
    
	// rozwija wezel drzewa o wszystkie mozliwe ruchy
    void expand(MCTSTreeNode* root);

	// wybiera wezel do symulacji
    MCTSTreeNode* select(MCTSTreeNode* root);

	// propaguje wyniki symulacji w gore drzewa
    void backpropagate(MCTSTreeNode* node, int delta_wins, int delta_visits);

	// wykonuje ruch
    void MakeMove(Board& board, char* ret, int moves_without_progress) override;
};
