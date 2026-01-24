#pragma once
#include <stdio.h>
#include <string.h>
#include "common.h"



class Player {
private:
	void buildNeighbourTabs() {
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

protected:
    Color color;

	int8_t Neighbours[32][4];
    int8_t Captures[32][4];
    uint32_t Rays[32][4];

public:
    Player(Color c) : color(c) {
		buildNeighbourTabs();
		buildRayTab();
	}

    virtual char* MakeMove(Board& board) = 0;
};


