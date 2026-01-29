#pragma once
#include <vector>
#include "common.h"
#include "moves_getter.h"

class Player {
private:
	// funckje pomocnicze do budowy tablic
	void buildCapturesTab() {
		const int8_t DEST_OFFSETS_CLOCK[4] = { -7, 9, 7, -9 };
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 4; j++) {
				int new_ind = i + DEST_OFFSETS_CLOCK[j];
				if (new_ind > 31 || new_ind < 0) {
					Captures[i][j] = -1;
					continue;
				}

				if (new_ind / 4 % 2 != i / 4 % 2) {
					Captures[i][j] = -1;
					continue;
				}
				Captures[i][j] = new_ind;
			}
		}
	}

	void buildNeighbourTab() {
		const int8_t WITH_OFFSETS_CLOCK[4] = { -4, 4, 3, -5 };

		for (int i = 0; i < 32; i++) {
			int offset = 1 - ((i / 4) % 2);
			for (int j = 0; j < 4; j++) {
				int new_ind = i + WITH_OFFSETS_CLOCK[j] + offset;
				if (new_ind > 31 || new_ind < 0) {
					Neighbours[i][j] = -1;
					continue;
				}

				if (new_ind / 4 % 2 == i / 4 % 2) {
					Neighbours[i][j] = -1;
					continue;
				}
				Neighbours[i][j] = new_ind;
			}
		}
	}

	void buildRayTab() {
		for (int i = 0; i < 32; i++) {
			for (int dir = 0; dir < 4; dir++) {

				uint32_t mask = 0;
				int current = i;

				while (true) {
					current = Neighbours[current][dir];
					if (current == -1) break;

					mask |= (1 << current);
				}

				Rays[i][dir] = mask;
			}
		}
	}

	// oddzielny obiekt do generowania legalnych
	MovesGetter moves_getter{ Neighbours, Captures };
protected:
	Color player_color;

	int8_t Neighbours[32][4];
    int8_t Captures[32][4];
    uint32_t Rays[32][4];


public:
    Player(Color c) : player_color(c) {
		buildCapturesTab();
		buildNeighbourTab();
		buildRayTab();

		moves_getter = MovesGetter(Neighbours, Captures);
	}

	// glowna funckaj klasy odpowidziealna za wykonanie tury gracza
    virtual void MakeMove(Board& board, char* ret, int moves_without_progress) = 0;


	std::vector<PossibleMove> getAllMoves(const Board& board, Color side_to_move) {
		return moves_getter.getAllMoves(board, side_to_move);
	}


	// funkcje pomocnicze do obslugi termianla
	bool fieldToChar(int idx, char* out) {

		int s = idx * 2 + (1 - (idx / 4) % 2);

		int f = 7 - s / 8;
		int r = s % 8;


		out[0] = char('a' + f);
		out[1] = char('1' + r);
		out[2] = '\0';
		return true;
	}

	bool charToField(char* sq, int& idx) {
		if (!sq) return false;

		char file = sq[0];
		char rank = sq[1];

		if (file < 'a' || file > 'h') return false;
		if (rank < '1' || rank > '8') return false;

		int f = file - 'a';
		int r = rank - '1';


		if ((f + r) % 2 == 1) {
			return false;
		}
		idx = (r + (7 - f) * 8) / 2;

		return true;
	}

	void moveToChar(std::vector<int> mv, bool is_capture, char* ret) {
		char delim = is_capture ? ':' : '-';
		int i = 0;
		for (int field : mv) {
			fieldToChar(field, ret + i * 3);
			ret[3 * i + 2] = delim;
			i++;
		}
		ret[3 * i - 1] = '\0';
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
};


