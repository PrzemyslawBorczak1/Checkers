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


        if (player_color == Color::WHITE) {
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

    bool isEmpty(int from, int to, int dir, uint32_t occ_total) {
        int iter = Neighbours[from][dir];
        while (iter != -1 && !(occ_total & (1 << iter)))
        {

            printf("iter %d\n", iter);
            if (iter == to) {
                return true;
            }
            iter = Neighbours[iter][dir];
        }
        printf("zajete pole %d\n", iter);
        return false;

    }

    bool perforomCaptureCheck(vector<int>& steps, Board board) {
        return false;
    }

   
    bool performNormalMoveCheck(vector<int>& steps, Board& board) {
        printf("Normal move \n");
        if (steps.size() != 2 || steps[0] == steps[1]) {
            printf("Bad move %d %d\n", steps[0], steps[1]);
            return false;
        }

        int dir = dirFromSteps(steps[0], steps[1]);
        if (dir < 0) {
            printf("Bad move no dir %d %d\n", steps[0], steps[1]);
            return false;
        }
		printf("Direction %d\n", dir);
        if(isEmpty(steps[0], steps[1], dir, 
            board.white_pawns | board.white_kings | board.black_pawns | board.black_kings)) {
            switch (player_color)
            {
            case Color::WHITE:
                if (board.white_pawns & (1 << steps[0])) {

                    printf("pionek step 0 %d \n", steps[0]);
                    if (dir == 0 || dir == 1) {
                        board.white_pawns &= ~(1 << steps[0]);
                        board.white_pawns |= (1 << steps[1]);
                        return true;
                    }
                    return false;
                }
                else if (board.white_kings & (1 << steps[0])) {
                    board.white_kings &= ~(1 << steps[0]);
                    board.white_kings |= (1 << steps[1]);
                    return true;
                }
                return false;
                break;
            case Color::BLACK:
                if (board.black_pawns & (1 << steps[0])) {
                    if (dir == 2 || dir == 3) {
                        board.black_pawns &= ~(1 << steps[0]);
                        board.black_pawns |= (1 << steps[1]);
                        return true;
                    }
                    return false;
                }
                else if (board.black_kings & (1 << steps[0])) {
                    board.black_kings &= ~(1 << steps[0]);
                    board.black_kings |= (1 << steps[1]);
                    return true;
                }
                break;
            }
		}
       
    
    }


public:
    HumanPlayer(Color c) : Player(c) {}

    char* MakeMove(Board& board)  {
        printf("\nHuman ruch\n");

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

            printf("ruch: %s\n", buf);
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



