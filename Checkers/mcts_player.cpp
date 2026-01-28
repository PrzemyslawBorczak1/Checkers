#include "mcts_player.h"

#include <cstdio>
#include <chrono>
#include <cmath>
#include <limits>

#include "mcts_kernel.cuh" 


inline const char* color_name(Color c) {
    return (c == Color::WHITE) ? "WHITE" : "BLACK";
}


// wyswietla zebrane statystyki z drzewa MCTS
void MCTSPlayer::printCollectedStats(const MCTSTreeNode* root, uint64_t counter) {
    if (!root) {
        printf("[MCTS] root == nullptr\n");
        return;
    }

    printf("VISITS=%llu WINS=%llu side to move=%s visits/second: %.3f Millions\n",
        (unsigned long long)root->visits,
        (unsigned long long)root->wins,
        color_name(root->side_to_move),
        (double)root->visits / (time_limit_sec * 1000000.0));


    if (root->children.empty()) {
        printf("No following moves found.\n");
        printf("=========================================================\n");
        return;
    }

    printf("\n# | move           | visits          | wins\n");
    printf("--+---------------+-----------------+-----------------\n");

    for (size_t i = 0; i < root->children.size(); ++i) {
        MCTSTreeNode* ch = root->children[i];

        if (!ch) {
            printf("%2zu| %-13s | %-15s | %-15s\n",
                i, "nullptr", "-", "-");
            continue;
        }

        unsigned long long v = (unsigned long long)ch->visits;
        unsigned long long w = (unsigned long long)ch->wins;

        char move[40];
        moveToChar(ch->possible_move.move, ch->possible_move.is_capture, move);

        printf("%2zu| %-13s | %15llu | %15llu\n",
            i, move, v, w);
    }


}


int MCTSPlayer::simulate(Board board, Color next_color, uint32_t seed, int moves_without_progress) {
    uint32_t ret = gpu.simulate(board, next_color, seed, moves_without_progress);
    return ret;

}

void MCTSPlayer::expand(MCTSTreeNode* root) {
    vector<PossibleMove> possible_moves = getAllMoves(root->possible_move.resulting_board, root->side_to_move);
    for (auto pm : possible_moves) {
        Color next_color = (root->side_to_move == Color::WHITE) ? Color::BLACK : Color::WHITE;
        MCTSTreeNode* child = new MCTSTreeNode(pm, next_color);
        child->parent = root;
        root->children.push_back(child);
    }
    root->is_expanded = true;
}

MCTSTreeNode* MCTSPlayer::select(MCTSTreeNode* root) {
    MCTSTreeNode* node = root;
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

        // jesli sa jakies nie odwiedzone dzieci to losowy
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

        uint64_t parent_visits = (node->visits > 0) ? node->visits : 0;


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
    }
}

void MCTSPlayer::backpropagate(MCTSTreeNode* node, int delta_wins, int delta_visits) {
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

void freeTree(MCTSTreeNode* n) {
    if (!n) return;
    for (auto* ch : n->children) freeTree(ch);
    delete n;
}

void MCTSPlayer::MakeMove(Board& board, char* ret, int moves_without_progress) {
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

        int wins = simulate(node->possible_move.resulting_board, node->side_to_move, seed_for_iter(base_seed, counter), moves_without_progress);
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

	freeTree(root);
}

