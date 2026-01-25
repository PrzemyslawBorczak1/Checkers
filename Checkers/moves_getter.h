#pragma once

#include <stdio.h>
#include <string.h>
#include <vector>
#include<iostream>
#include "common.h"
using namespace std;

struct PossibleMove {
	vector<int> move;
	Board resulting_board;
};

struct RecBoard {
	uint32_t player_pawns;
	uint32_t player_kings;
	uint32_t enemy_pawns;
	uint32_t enemy_kings;
	uint32_t occ_player() {
		return player_pawns | player_kings;
	}
	uint32_t occ_enemy() {
		return enemy_pawns | enemy_kings;
	}
	uint32_t occ_total() {
		return occ_player() | occ_enemy();
	}
};




class MovesGetter {

public:	
	MovesGetter(const int8_t(&neigh)[32][4], const int8_t(&caps)[32][4])
	{
		copy(&neigh[0][0], &neigh[0][0] + 32 * 4, &Neighbours[0][0]);
		copy(&caps[0][0], &caps[0][0] + 32 * 4, &Captures[0][0]);
	}

private:
	vector<PossibleMove> moves;
	bool is_capture = false;
	Color side_to_move;

	int8_t Neighbours[32][4];
	int8_t Captures[32][4];


	RecBoard rec_board;
	Board active_board;


	void addPawnMove(vector<int> move) {
		PossibleMove pm;
		pm.move = move;

		Board res_board = active_board;
		int final_to = move.back();
		switch (side_to_move)
		{
		case Color::WHITE:
			res_board.white_pawns &= ~(1 << move[0]);

			if ( (final_to % 4) == 3 && ((final_to / 4) % 2) == 0) {
				res_board.white_kings |= 1 << final_to;
			}
			else {
				res_board.white_pawns |= (1 << move.back());
			}
			break;
		case Color::BLACK:
			if ((final_to % 4) == 0 && ((final_to / 4) % 2) == 1) {
				res_board.black_kings |= 1 << final_to;
			}
			else {
				res_board.black_pawns |= (1 << move.back());
			}
			break;
		}


		pm.resulting_board = res_board;
		moves.push_back(pm);
	}

	void addPawnCapture(vector<int> move, uint32_t occ_total) {
		PossibleMove pm;
		pm.move = move;

		Board res_board;
		res_board.white_pawns = active_board.white_pawns & occ_total;
		res_board.white_kings = active_board.white_kings & occ_total;
		res_board.black_pawns = active_board.black_pawns & occ_total;
		res_board.black_kings = active_board.black_kings & occ_total;

		int final_to = move.back();
		switch (side_to_move)
		{
		case Color::WHITE:
			res_board.white_pawns &= ~(1 << move[0]);

			if ((final_to % 4) == 3 && ((final_to / 4) % 2) == 0) {
				res_board.white_kings |= 1 << final_to;
			}
			else {
				res_board.white_pawns |= (1 << move.back());
			}
			break;
		case Color::BLACK:
			if ((final_to % 4) == 0 && ((final_to / 4) % 2) == 1) {
				res_board.black_kings |= 1 << final_to;
			}
			else {
				res_board.black_pawns |= (1 << move.back());
			}
			break;
		}

		pm.resulting_board = res_board;

		moves.push_back(pm);
	}

	void addKingMove(vector<int> move, uint32_t occ_total) {
		PossibleMove pm;
		pm.move = move;

		Board res_board;
		res_board.white_pawns = active_board.white_pawns & occ_total;
		res_board.white_kings = active_board.white_kings & occ_total;
		res_board.black_pawns = active_board.black_pawns & occ_total;
		res_board.black_kings = active_board.black_kings & occ_total;


		switch (side_to_move)
		{
		case Color::WHITE:
			res_board.white_kings &= ~(1 << move[0]);
			res_board.white_kings |= (1 << move.back());
			break;
		case Color::BLACK:
			res_board.black_kings &= ~(1 << move[0]);
			res_board.black_kings |= (1 << move.back());
			break;
		}

		pm.resulting_board = res_board;
		moves.push_back(pm);
	}


	void show() {
		printf("show\n");
		for (const auto& mv : moves) {
			std::cout << "[ ";
			for (int x : mv.move) std::cout << x << ' ';
			std::cout << "]\n";
			printf("old board:\n");
			printBoard(active_board);
			printf("new board:\n");
			printBoard(mv.resulting_board);
			printf("\n\n\n");
		}
	}

	void setIsCaptureTrue() {
		if (is_capture == false) {
			moves.clear();
			is_capture = true;
		}
	}

	
	// dodaje ruchy pionka z pola from
	void pawnMoves(RecBoard rec_board, int from) {
		int to;
		if (rec_board.player_pawns & (1 << from)) {
			if (side_to_move == Color::WHITE) {
				to = Neighbours[from][0];
			}
			else {
				to = Neighbours[from][2];
			}
			if (to != -1) {
				if (!(rec_board.occ_total() & (1 << to))) {
					addPawnMove({ from, to });
				}
			}

			if (side_to_move == Color::WHITE) {
				to = Neighbours[from][1];
			}
			else {
				to = Neighbours[from][3];
			}

			if (to != -1) {
				if (!(rec_board.occ_total() & (1 << to))) {
					addPawnMove({ from, to });
				}
			}
		}
	}

	// dodaje bicia pionka z pola from
	void pawnCaptures(uint32_t occ_enemy, uint32_t occ_total, int from, vector<int>& move) {

		bool continued = false;
		for (int i = 0; i < 4; i++) {
			int with = Neighbours[from][i];
			int to = Captures[from][i];
			if (with == -1 || to == -1)
				continue;
			if (occ_enemy & (1 << with)) {
				if (!(occ_total & (1 << to))) {
					continued = true;
					setIsCaptureTrue();
					move.push_back(to);
					pawnCaptures(occ_enemy & ~(1 << with), occ_total & ~(1 << with), to, move);
					move.pop_back();
				}
			}
		}

		if (!continued && move.size() > 1) {
			addPawnCapture(move, occ_total);
		}
	}


	// Zwraca -index przeciwnika ktorego mozna zbic na lini zaczynajacej sie od index w kierunku dir. Pole from nie jest sprawdzane.
	// W przypadku braku bicia zwraca ilosc pol mozliwych do przejscia
	int checkLine(uint32_t occ_enemy, uint32_t occ_total, int from, int dir) {
		int counter = 0;
		int next = Neighbours[from][dir];
		while (next != -1)
		{
			if (occ_enemy & (1 << next) && Neighbours[next][dir] != -1 && !(occ_total & (1 << Neighbours[next][dir]))) {
				//printf("Check line from %d dir %d found enemy on %d\n", index, dir, next);
				return -next;
			}
			if (occ_total & (1 << next)) {
				break;
			}
			next = Neighbours[next][dir];
			counter++;

		}
		//printf("Check line from %d dir %d returns %d\n", index, dir, counter);
		return counter;
	}

	// dodaje ruchy damka po lini w kierunku dir od from (from nie jest dodawne)
	void addKingFinal(vector<int>& move, uint32_t occ_total, int from, int amount, int dir) {
		//printf("King final from %d amount %d dir %d\n", from, amount, dir);
		int next = from;
		for (int i = 0; i < amount; i++) {
			next = Neighbours[next][dir];
			move.push_back(next);
			addKingMove(move, occ_total);
			move.pop_back();
		}
	}

	// dodaje ruchy damka z pola from
	void kingMoves(uint32_t occ_enemy, uint32_t occ_total, int from, vector<int>& move) {
		int lines[4];
		for (int dir = 0; dir < 4; dir++) {
			lines[dir] = checkLine(occ_enemy, occ_total, from, dir);
			//printf("Line first %d: %d\n", i, lines[i]);
			if (lines[dir] < 0) {
				setIsCaptureTrue();
				kingCaptures(occ_enemy & ~(1 << -lines[dir]), occ_total & ~(1 << -lines[dir] | 1 << from), -lines[dir], dir, move);

			}
		}

		//printf("Is capture: %d\n", is_capture);
		if (!is_capture) {
			for (int dir = 0; dir < 4; dir++) {
				//printf("Line %d: %d\n", i, lines[i]);
				addKingFinal(move, occ_total, from, lines[dir], dir);
			}
		}
	}

	// dodaje rozgaleznie bicia damka zaczynajace sie na lini dir
	void kingCaptures(uint32_t occ_enemy, uint32_t occ_total, int from, int dir, vector<int>& move) {
		//printf("King captures start %d\n occc_total:\n", index);
		//print_int(occ_total);

		int next = Neighbours[from][dir];
		bool continued = false;
		while (next != -1) {
			//printf("King capture checking dir %d next %d\n",  dir, next);

			// bicie na glownej linii
			if (occ_enemy & (1 << next)) {
				if (Neighbours[next][dir] != -1 && !(occ_total & (1 << Neighbours[next][dir]))) {

					continued = true;
					int enemy_index = next;
					//printf("Found capture  in main line next: %d  dir %d enemy %d\n", next, dir, enemy_index);
					move.push_back(Neighbours[next][(dir + 2) % 4]);
					kingCaptures(occ_enemy & ~(1 << enemy_index), occ_total & ~(1 << enemy_index), next, dir, move);
					move.pop_back();

					break;
				}
			}

			if (occ_total & (1 << next)) {
				break;
			}


			int line = 67;
			// rozgalezienie 1
			line = checkLine(occ_enemy, occ_total, next, (dir + 1) % 4);
			if (line < 0) {
				continued = true;
				int enemy_index = -line;
				//printf("Found capture in  line next: %d  dir %d enemy %d\n", next, dir, enemy_index);
				move.push_back(next);
				kingCaptures(occ_enemy & ~(1 << enemy_index), occ_total & ~(1 << enemy_index), enemy_index, (dir + 1) % 4, move);
				move.pop_back();
			}

			// rozgalezienie 2
			line = checkLine(occ_enemy, occ_total, next, (dir + 3) % 4);
			if (line < 0) {
				continued = true;
				int enemy_index = -line;
				//printf("Found capture in  line next: %d  dir %d enemy %d\n", next, dir, enemy_index);
				move.push_back(next);
				kingCaptures(occ_enemy & ~(1 << enemy_index), occ_total & ~(1 << enemy_index), enemy_index, (dir + 3) % 4, move);
				move.pop_back();
			}


			next = Neighbours[next][dir];

		}
		// brak dalszych rozgalezien
		if (!continued) {
			//printf("Final from %d dir %d\n", index, dir);
			addKingFinal(move, occ_total, from, checkLine(occ_enemy, occ_total, from, dir), dir);
		}
	}

public:
	// glowna funkcja - zwraca wszystkie mozliwe ruchy dla danej pozycji i koloru do ruchu
	vector<PossibleMove> getAllMoves(const Board board, Color side_to_move_f) {

		active_board = board;
		moves.clear();
		is_capture = false;
		side_to_move = side_to_move_f;

		RecBoard rec_board;
		switch (side_to_move) {
		case Color::WHITE:
			rec_board.player_pawns = active_board.white_pawns;
			rec_board.player_kings = active_board.white_kings;
			rec_board.enemy_pawns = active_board.black_pawns;
			rec_board.enemy_kings = active_board.black_kings;
			break;
		case Color::BLACK:
			rec_board.player_pawns = active_board.black_pawns;
			rec_board.player_kings = active_board.black_kings;
			rec_board.enemy_pawns = active_board.white_pawns;
			rec_board.enemy_kings = active_board.white_kings;
			break;
		}

		for (int i = 0; i < 32; i++) {

			if (rec_board.player_pawns & (1 << i)) {
				// ruchy pionkow
				if (!is_capture)
					pawnMoves(rec_board, i);

				// bicia pionkami
				vector<int> move;
				move.push_back(i);
				pawnCaptures(rec_board.occ_enemy(), rec_board.occ_total() & ~(1 << i), i, move);
				move.pop_back();
			}

			// ruchy damkami
			if (rec_board.player_kings & (1 << i)) {
				vector<int> move;
				move.push_back(i);
				kingMoves(rec_board.occ_enemy(), rec_board.occ_total() & ~(1 << i), i, move);
				move.pop_back();
			}
		}


		return moves;
	}

};