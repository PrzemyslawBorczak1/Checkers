#pragma once
#include <stdio.h>
#include "common.h"
#include "player.h"
#include "moves_getter.h" 
#include "GPU.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include "mcts_kernel.cuh"
#define EXPLORE_CONSTANT 1.41



#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>


using namespace std;

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

    // todo ususac 2 nastepen
    static inline const char* color_name(Color c) {
        return (c == Color::WHITE) ? "WHITE" : "BLACK";
    }



    void printCollectedStats(MCTSTreeNode* root, uint64_t counter) {
        if (!root) {
            printf("[MCTS] root == nullptr\n");
            return;
        }


        printf("VISITS=%d WINS=%d side to move = %s vist/second: %f Milions\n",
            root->visits, root->wins, color_name(root->side_to_move),
            (double)root->visits / (time_limit_sec * 1000000));

        if (root->children.empty()) {
            printf("No following moves found.\n");
            printf("=========================================================\n");
            return;
        }

        printf("\n# | move           | visits     | wins       | winrate(W)\n");
        printf("--+---------------+------------+------------+-----------\n");


        for (size_t i = 0; i < root->children.size(); ++i) {
            MCTSTreeNode* ch = root->children[i];

            if (!ch) {
                printf("%2zu| %-13s | %-10s | %-10s | %-9s\n",
                    i, "nullptr", "-", "-", "-");
                continue;
            }

            double wr = (root->side_to_move == Color::WHITE) ? ((double)ch->wins / (double)ch->visits) : 1.0 - (double)ch->wins / (double)ch->visits;

            char move[40];
            moveToChar(ch->possible_move.move, ch->possible_move.is_capture, move);
            printf("%2zu| %s | %10d | %10d | %9.4f\n",
                i, move, ch->visits, ch->wins, wr);
        }

	}


    int simulate(Board board, Color next_color, uint32_t seed, int moves_without_progress) {
        uint32_t ret = gpu.simulate(board, next_color, seed, moves_without_progress);
        return ret;
       
    }

    void expand(MCTSTreeNode* root) {
        vector<PossibleMove> possible_moves = getAllMoves(root->possible_move.resulting_board, root->side_to_move);
        for (auto pm : possible_moves) {
            Color next_color = (root->side_to_move == Color::WHITE) ? Color::BLACK : Color::WHITE;
            MCTSTreeNode* child = new MCTSTreeNode(pm, next_color);
			child->parent = root;
            root->children.push_back(child);
        }
        root->is_expanded = true;
        
    }


    MCTSTreeNode* select(MCTSTreeNode* root) {
        MCTSTreeNode* node = root;
		//printf("\n\n\n\n\n\n\nNew selection from root:");
        //debug_print_children(node);
        while (true) {

            if (!node->is_expanded) {
                if (node->visits != 0)
                    expand(node);
                else
                    return node;
            }
            
            if (node->children.empty() && node->is_expanded) {
                return node;
            }

            // jelsi sa jakeis nie odwiedzone to losowy
            std::vector<MCTSTreeNode*> unvisited;
            unvisited.reserve(node->children.size());
            for (MCTSTreeNode* child : node->children) {
                if (child && child->visits == 0)
                    unvisited.push_back(child);
            }

            if (!unvisited.empty()) {
                int idx = rand() % unvisited.size();
                return unvisited[idx];
            }

			// wybor dziecka z najlepszym UCT
            MCTSTreeNode* best_child = nullptr;
            double best_score = -std::numeric_limits<double>::infinity();

            Color player = node->side_to_move;

            uint64_t parent_visits = (node->visits > 0) ? (double)node->visits : 1.0;


            for (MCTSTreeNode* child : node->children) {
                if (!child) continue;

                double win_ratio = (double)child->wins / (double)child->visits;
                double exploit = (player == Color::WHITE) ? win_ratio : (1.0 - win_ratio);

                double explore = EXPLORE_CONSTANT * sqrt(log(parent_visits) / (double)child->visits);
                double uct = exploit + explore;

                if (uct > best_score) {
                    best_score = uct;
                    best_child = child;
                }
            }

            if (!best_child)
                return nullptr;

            node = best_child;

          //  debug_print_children(node);
            
        }
    }

    void backpropagate(MCTSTreeNode* node, int delta_wins, int delta_visits) {
        MCTSTreeNode* current = node;
        for (auto* cur = node; cur != nullptr; cur = cur->parent) {
            cur->visits += delta_visits;
            cur->wins += delta_wins;  
        }
	}

   uint32_t mix32(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    uint32_t seed_for_iter(uint32_t base_seed, uint32_t iter) {
        uint32_t s = base_seed ^ (iter * 0x9e3779b9U);
        s = mix32(s);
        return s ? s : 1;
    }

    MCTSTreeNode* findBest(MCTSTreeNode* root) {
        if (!root || root->children.empty()) return nullptr;

        MCTSTreeNode* best = nullptr;
        double best_score = -1.0;

        for (MCTSTreeNode* ch : root->children) {
            if (!ch || ch->visits == 0) continue;

            double wr_white = (double)ch->wins / (double)ch->visits;

            double score = (root->side_to_move == Color::WHITE) ? wr_white : (1.0 - wr_white);

            if (score > best_score) {
                best_score = score;
                best = ch;
            }
        }

        return best;
    }



    void MakeMove(Board& board, char* ret, int moves_without_progress) override {
        printf("MCTS Player making move:\n");
		PossibleMove root_pm = { {}, board };
		root = new MCTSTreeNode(root_pm, player_color);
        uint64_t base_seed =
            (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        

        auto start = std::chrono::steady_clock::now();
		int counter = 0;
        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(time_limit_sec * 1000 - 50)) {
            MCTSTreeNode* node = select(root);
            if (!node) 
                continue;
            
            int wins = simulate(node->possible_move.resulting_board, node->side_to_move, seed_for_iter(base_seed,counter), moves_without_progress);
            backpropagate(node, wins, BLOCKS * THREADS);
            counter++;
        }

        printf("\n================= SUMMARY =================\n");
        printCollectedStats(root, counter);
		MCTSTreeNode* best_move_node = findBest(root);
        if (!best_move_node) {
            printf("MCTSPlayer: No best move found! Returning null move.\n");
			ret[0] = '\0';
            return;
		}
        moveToChar(best_move_node->possible_move.move, best_move_node->possible_move.is_capture, ret);
		printf("Best move selected: %s\n", ret);


        printf("=========================================================\n\n");
		board = best_move_node->possible_move.resulting_board;
    
    }

};
