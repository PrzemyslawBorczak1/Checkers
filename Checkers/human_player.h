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
		uint32_t occ_total = board.white_pawns | board.white_kings | board.black_pawns | board.black_kings;
        
		uint32_t player_pawns, player_kings, occ_opponenet;


        if (color == Color::WHITE) {
            player_kings = board.white_kings;
			player_pawns = board.white_pawns;
            occ_opponenet = board.black_pawns | board.black_kings;
        }
        else {
			player_kings = board.black_kings;
            player_pawns = board.black_pawns;
			occ_opponenet = board.white_pawns | board.white_kings;
        }
            

		for (int i = 0; i < 32; i++) {
            if (player_kings & (1 << i)) {
                for (int j = 0; j < 4; j++) {
                    if(countEmptySquaresCPU(occ_total, occ_opponenet, i, j) < 0)
						return true;
                }
            }


            if(player_pawns & (1 << i)) {
                for (int j = 0; j < 4; j++) {
                    uint32_t with = Neighbours[i][j];
                    uint32_t to = Captures[i][j];
                    if (with == -1 || to == -1) {
                        continue;
                    }

                    if ((occ_opponenet & (1 << with))) {
                        if (!(occ_total & (1 << to))) {
                            return true;
                        }
                    }
                }
			}
		}
    
        
		return false;
    
    }

    bool squareToIndex(char* sq, int& idx) {
        idx = sq[0] + sq[1];
        return true;
    }

    bool parseMove(char* move_str, vector<int>& steps, char delim) {
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

    bool perforomCaptureCheck(vector<int>& steps, Board board) {
        return false;
    }
    bool performNormalMoveCheck(vector<int>& steps, Board board) {
		return false;
	}


public:
    HumanPlayer(Color c) : Player(c) {}

    char* MakeMove(Board& board)  {
        printf("\nHuman ruch");

		bool is_capture = isCaptureOnBoard(board);
        char delim;
        if(is_capture) {
            printf("Masz bicie!\n");
			delim =  ':';
		}
        else {
            delim = '-';
        }

        vector<int> steps;

        char buf[256];


        while (true)
        {
            printf("Podaj ruch: ");
            fflush(stdout);

            if (!fgets(buf, (int)sizeof(buf), stdin)) {
                return "\0";
            }
            size_t n = strlen(buf);
            if (n > 0 && buf[n - 1] == '\n')
                buf[n - 1] = '\0';

            if (!parseMove(buf, steps, delim)) {
                printf("Nie odpowiedni ruch\n");
                continue;
            }

            printf("ruch:\n");
            for (int i = 0; i < (int)steps.size(); i++) {
                printf("%d,\n", steps[i]);
            }

            if (is_capture) {
                if (perforomCaptureCheck(steps, board)) {
					break;
                }
            }
            else {
                if (performNormalMoveCheck(steps, board)) {
                    break;
                }
            }

        }

        return buf;
    }
};



