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

    void debug_print_children(MCTSTreeNode* node) {
        if (!node) {
            std::cout << "[DEBUG] node == nullptr\n";
            return;
        }

        std::cout << "\n=== NODE DEBUG ===\n";
        std::cout << "node: " << ((int)node % 100)
            << " side_to_move: " << color_name(node->side_to_move)
            << " visits: " << node->visits
            << " wins: " << node->wins
            << " is_expanded: " << (node->is_expanded ? "true" : "false")
            << " children: " << node->children.size()
            << "\n";

        if (node->children.empty()) {
            std::cout << "no children\n";
            return;
        }

        for (size_t i = 0; i < node->children.size(); ++i) {
            MCTSTreeNode* ch = node->children[i];

            std::cout << "  [" << i << "] ";

            if (!ch) {
                std::cout << "nullptr\n";
                continue;
            }

            double wr = (ch->visits > 0) ? (double)ch->wins / (double)ch->visits : 0.0;

            std::cout << "ptr=" << ((int)ch % 100)
                << " v=" << ch->visits
                << " w=" << ch->wins
                << " win_ratio=" << wr
                << " expanded=" << (ch->is_expanded ? "true" : "false")
                << " kids=" << ch->children.size()
                << "\n";
			printBoard(ch->possible_move.resulting_board);
        }
    }

    void print_root_children_summary(MCTSTreeNode* root) {
        if (!root) {
            printf("[MCTS] root == nullptr\n");
            return;
        }

        printf("\n================= ROOT CHILDREN SUMMARY =================\n");
        printf("Root: visits=%d wins=%d side_to_move=%s expanded=%s children=%zu\n",
            root->visits, root->wins, color_name(root->side_to_move),
            root->is_expanded ? "true" : "false",
            root->children.size());

        if (root->children.empty()) {
            printf("No children.\n");
            printf("=========================================================\n");
            return;
        }

        printf("\n# | ptr           | visits     | wins       | winrate(W)\n");
        printf("--+---------------+------------+------------+-----------\n");

        int best_by_visits_i = -1;
        int best_visits = -1;

        int best_by_wr_i = -1;
        double best_wr = -1.0;

        for (size_t i = 0; i < root->children.size(); ++i) {
            MCTSTreeNode* ch = root->children[i];

            if (!ch) {
                printf("%2zu| %-13s | %-10s | %-10s | %-9s\n",
                    i, "nullptr", "-", "-", "-");
                continue;
            }

            double wr = (ch->visits > 0) ? ((double)ch->wins / (double)ch->visits) : 0.0;

            printf("%2zu| %p | %10d | %10d | %9.4f\n",
                i, (void*)ch, ch->visits, ch->wins, wr);

            if (ch->visits > best_visits) {
                best_visits = ch->visits;
                best_by_visits_i = (int)i;
            }

            // winrate ma sens dopiero jak visits>0
            if (ch->visits > 0 && wr > best_wr) {
                best_wr = wr;
                best_by_wr_i = (int)i;
            }
        }

        printf("\nBest by visits : child #%d (visits=%d)\n", best_by_visits_i, best_visits);
        if (best_by_wr_i >= 0)
            printf("Best winrate(W): child #%d (wr=%.4f)\n", best_by_wr_i, best_wr);

        printf("=========================================================\n\n");
    }


    int simulate(Board board, Color next_color, uint32_t seed) {
        uint32_t ret = gpu.simulate(board, next_color, seed);
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

    MCTSTreeNode* best_child_by_winrate(MCTSTreeNode* root) {
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

    char* MakeMove(Board& board) override {
        printf("MCTS Player making move:\n");
		PossibleMove root_pm = { {}, board };
		root = new MCTSTreeNode(root_pm, player_color);
        uint64_t base_seed =
            (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();


        auto start = std::chrono::steady_clock::now();
		int counter = 0;
        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(time_limit_sec * 1000 - 50)) {
            MCTSTreeNode* node = select(root);
            if (!node) continue;
            
            int wins = simulate(node->possible_move.resulting_board, node->side_to_move, seed_for_iter(base_seed,counter));
            backpropagate(node, wins, BLOCKS * THREADS);
            counter++;

        }


		printf("MCTS selection complete. Choosing best move...\n");

		printf("Visiting counter: %d root children: %d  vist/second: %fM\n",counter, (int)root->visits, (double)root->visits / (time_limit_sec * 1000000));
        print_root_children_summary(root);
		MCTSTreeNode* best_move_node = best_child_by_winrate(root);
        if (!best_move_node) {
            printf("MCTSPlayer: No best move found! Returning null move.\n");
            return "\0";
		}
		printf("Best move selected: %d\n", (int)best_move_node);
    
        return "\0";
    }
};
