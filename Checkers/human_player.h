#pragma once
#pragma message("BUILD HIT: " __FILE__)

#include <iostream>
#include <vector>     
#include <cstdint>
#include "common.h"
#include "player.h"

using namespace std;





class HumanPlayer : public Player {
private:
    int8_t countEmptySquaresCPU(
        uint32_t occ_total,
        uint32_t occ_enemy,
        uint8_t index,
        uint8_t dir
    ) {
        int8_t current = Neighbours[index][dir];
        int8_t empty_count = 0;

        while (current != -1) {
            if (occ_total & (1u << current)) {
                if (occ_enemy & (1u << current)) {
                    int8_t landing_sq = Neighbours[current][dir];

                    if (landing_sq != -1 && !(occ_total & (1u << landing_sq))) {
                        return -current;
                    }
                }
                return empty_count;
            }
            empty_count++;
            current = Neighbours[current][dir];
        }
        return empty_count;
    }


    bool isCaptureOnBoard(Board& board) {
		getAllMoves(board, player_color);
        for (auto pm : getAllMoves(board, player_color)) {
            if (pm.is_capture) {
                return true;
            }
        }
		return false;
    
    }

    bool squareToIndex(char* sq, int& idx) {
        if (!sq) return false;

        char file = sq[0];
        char rank = sq[1];

        if (file < 'a' || file > 'h') return false;
        if (rank < '1' || rank > '8') return false;

        int f = file - 'a';  
        int r = rank - '1';  


        if((f + r) % 2 == 1) {
            return false;
		}
        idx = (r + (7 - f) * 8) / 2;

        return true;
    }

    bool parseMove(char* move_str, vector<int>& steps, char delim) {
		printf("Parsing move: %s\n", move_str);    
        char* current = move_str;
        steps.clear();

        while (current != nullptr && *current != '\0') {
            char* next_delim = strchr(current, delim);

            char buffer[3] = { 0 };

            size_t len = (next_delim != nullptr) ? (next_delim - current) : strlen(current);

            if (len == 0 || len >= sizeof(buffer)) {
                return false;
            }

            strncpy(buffer, current, len);
            buffer[len] = '\0';

            int idx = 0;
            if (!squareToIndex(buffer, idx)) {
                return false; 
            }

            steps.push_back(static_cast<int>(idx));

            if (next_delim != nullptr) {
                current = next_delim + 1;
            }
            else {
                current = nullptr;
            }
        }

        return steps.size() >= 2;
    }

    int dirFromSteps(int s1, int s2) {

        s1 = s1 * 2 + (1 - (s1 / 4) % 2);
        s2 = s2 * 2 + (1 - (s2 / 4) % 2);

        int x1 = s1 / 8, y1 = s1 % 8;
        int x2 = s2 / 8, y2 = s2 % 8;

        int dx = x2 - x1;
        int dy = y2 - y1;

        if (dx == 0 && dy == 0) return -1;

        if (dx < 0 && dy > 0) return 0;
        if (dx > 0 && dy > 0) return 1;
        if (dx > 0 && dy < 0) return 2;
        if (dx < 0 && dy < 0) return 3;

        return -1;
    }

    vector<int> normalizeMove( vector<int>& steps)
    {
        const int n = (int)steps.size();
        if (n <= 2) return steps; 

        vector<int> out;
        out.reserve(n);
        out.push_back(steps[0]); 

        for (int i = 1; i < n - 1; ++i) {
            int d1 = dirFromSteps(steps[i - 1], steps[i]);
            int d2 = dirFromSteps(steps[i], steps[i + 1]);

            if (d1 == d2) continue;

            out.push_back(steps[i]);
        }

        out.push_back(steps[n - 1]); 
        return out;
    }

    bool moveEqualsNormalized( vector<int>& a, vector<int>& b)
    {
		char buffa[40];
		char buffb[40];
		moveToChar(a, false, buffa);
		moveToChar(b, false, buffb);
		printf("Comparing moves: %s  against  %s\n", buffa, buffb);


        vector<int> na = normalizeMove(a);
        vector<int> nb = normalizeMove(b);
        return na == nb;
    }

    PossibleMove* findMoveNormalized(vector<PossibleMove>& pm, vector<int>& m)
    {
        vector<int> nm = normalizeMove(m);

        for (auto& p : pm) {
            if (normalizeMove(p.move) == nm) {
                return &p;
            }
        }
        return nullptr;
    }

    void printMoveVector(vector<PossibleMove> pm) {
        for (auto p: pm) {
            
            char buff[40];
			moveToChar(p.move, p.is_capture, buff);
			printf("Move: %s\n", buff);
		}
        printf("\n");
	}


public:
    HumanPlayer(Color c) : Player(c) {}

    void MakeMove(Board& board, char* ret, int moves_without_progress)  {
		ret[0] = '\0';
		char buf[100];
		buf[0] = '\0';
        printf("\nHuman ruch\n");

		vector<PossibleMove> possible_moves = getAllMoves(board, player_color);
        if (possible_moves.empty()) {
            printf("Brak mozliwych ruchow!\n");
			buf[0] = '\0';
			return;
        }
		bool is_capture = false;
        if (possible_moves[0].is_capture) {
			is_capture = true;
            printf("Masz bicie!\n");
        }

		char delim = is_capture ? ':' : '-';

        vector<int> steps;



        while (true)
        {
			buf[0] = '\0';
			printf("mozliwe ruchy:\n");
			printMoveVector(possible_moves);
            printf("Podaj ruch: ");
            fflush(stdout);

            if (!fgets(buf, (int)sizeof(buf), stdin)) {
				ret[0] = '\0';
                return;
            }
            size_t n = strlen(buf);
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
                buf[--n] = '\0';
            }

            if (!parseMove(buf, steps, delim)) {
                printf("Nie odpowiedni format ruchu\n");
                continue;
            }

            printf("ruch: %s\n", buf);
            printf("vec: %s\n", buf);
            for (int i = 0; i < (int)steps.size(); i++) {
                printf(" %d ", steps[i]);
            }

			auto fn = findMoveNormalized(possible_moves, steps);
            if(fn == nullptr) {
                printf("Ruch nie istnieje w dozwolonych\n");
                
                continue;
            }
            
			board = fn->resulting_board;
			printf("Ruch zaakceptowany\n");
            printf("ret string: %s\n", buf);
      
			strcat(ret, buf);
			return;

        }
    }
};



