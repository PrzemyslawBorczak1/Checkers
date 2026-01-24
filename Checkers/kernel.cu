#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdint>
#include <cstring>

#include "cpu_common.h"



__device__ __constant__  int8_t NEIGHBOURS[32][4];
__device__ __constant__  int8_t CAPTURES[32][4];
__device__ __constant__  uint32_t RAYS[32][4];


int8_t Neighbours[32][4];
int8_t Captures[32][4];
uint32_t Rays[32][4];


// move_packed: next_enemy | dir | type | from | with | to
# define MOVE_TYPE_NORMAL 0u
# define MOVE_TYPE_PAWN_CAPTURE 1u
# define MOVE_TYPE_KING_FINAL_CAPTURE 3u
# define MOVE_TYPE_KING_BRANCHING_CAPTURE 2u

__device__ __forceinline__ void unpackMove(
	uint32_t packed,
	uint8_t& from,
	uint8_t& with,
	uint8_t& to,
	uint8_t& type,
	uint8_t& dir,
	uint8_t& next_enemy
) {
	from = (uint8_t)(packed & 31u);
	with = (uint8_t)((packed >> 5) & 31u);
	to = (uint8_t)((packed >> 10) & 31u);
	type = (uint8_t)((packed >> 15) & 31u);
	dir = (uint8_t)((packed >> 20) & 31u);
	next_enemy = (uint8_t)((packed >> 25) & 31u);
}


__device__ __forceinline__ void addKingBranchingCapture(
	uint32_t& move_packed,
	uint32_t& seed,
	uint8_t& count,

	uint8_t from,
	uint8_t with,
	uint8_t to,

	uint8_t dir,
	uint8_t next_enemy
) {
	//printf("Add king branching capture %d %d %d %d %d count %d\n", from, with, to, dir, next_enemy, count);
	count++;
	uint32_t x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	seed = x;
	if ((x % (uint32_t)count) == 0u) {
		move_packed = (uint32_t)0 |
			((next_enemy & 31u) << 25) |
			((dir & 31u) << 20) |
			MOVE_TYPE_KING_BRANCHING_CAPTURE << 15 |
			((to & 31u) << 10) |
			((with & 31u) << 5) |
			(from & 31u);
	}
}

__device__ __forceinline__ void addKingFinalCapture(
	uint32_t& move_packed,
	uint32_t& seed,
	uint8_t& count,

	uint8_t from,
	uint8_t with,
	uint8_t to
) {
	//printf("Add king final capture %d %d %d  count %d\n", from, with, to, count);
	count++;
	uint32_t x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	seed = x;
	if ((x % (uint32_t)count) == 0u) {
		move_packed = (uint32_t)0 |
			MOVE_TYPE_KING_FINAL_CAPTURE << 15 |
			((to & 31u) << 10) |
			((with & 31u) << 5) |
			(from & 31u);
	}
}

__device__ __forceinline__ void addPawnCapture(
	uint32_t& move_packed,
	uint32_t& seed,
	uint8_t& count,

	uint8_t from,
	uint8_t with,
	uint8_t to
) {
	//printf("Add pawn capture %d %d %d  count %d\n", from, with, to, count);
	count++;
	uint32_t x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	seed = x;
	if ((x % (uint32_t)count) == 0u) {
		move_packed = (uint32_t)0 |
			MOVE_TYPE_PAWN_CAPTURE << 15 |
			((to & 31u) << 10) |
			((with & 31u) << 5) |
			(from & 31u);
	}
}
__device__ __forceinline__ void addNormalMove(
	uint32_t& move_packed,
	uint32_t& seed,
	uint8_t& count,

	uint8_t from,
	uint8_t to
) {
	//printf("Add new normal move %d %d  count %d\n", from, to, count);
	count++;
	uint32_t x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	seed = x;
	if ((x % (uint32_t)count) == 0u) {
		move_packed = (uint32_t)0 |
			MOVE_TYPE_NORMAL << 15 |
			((to & 31u) << 10) |
			(from & 31u);
	}
}





// zwraca liczbe wolnych pol w danym kierunku
// lub -index przeciwnika w przypadku bicia (nigdy 0)
__device__ __forceinline__ int8_t countEmptySquares(
	uint32_t occ_total,
	uint32_t occ_enemy,
	uint8_t index,
	uint8_t dir
) {
	uint32_t ray = RAYS[index][dir];
	uint32_t blockers = ray & occ_total;

	// popc ilosc bitow == 1
	if (blockers == 0) {
		return (int8_t)__popc(ray);
	}

	// pierwsza przeszkoda (index bomaski u32)
	// ffs liczba 0 od najmniej znaczącego bitu
	// clz liczba 0 od najbardziej znaczącego bitu
	int blocker_idx;
	if (dir == 1 || dir == 2) {
		blocker_idx = __ffs(blockers) - 1;
	}
	else {
		blocker_idx = 31 - __clz(blockers);
	}

	// czy jest bicie
	if (occ_enemy & (1u << blocker_idx)) {
		int8_t landing_sq = NEIGHBOURS[blocker_idx][dir];

		if (landing_sq != -1 && !(occ_total & (1u << landing_sq))) {
			return -blocker_idx;
		}
	}


	// pola przed przeszkodą
	// -1 tworzy maskę z jedynkami przed przeszkodą
	if (dir == 1 || dir == 2) {
		return (int8_t)__popc(ray & ((1u << blocker_idx) - 1));
	}
	else {
		uint32_t mask_above = (blocker_idx == 31) ? 0 : (0xFFFFFFFFu << (blocker_idx + 1));
		return (int8_t)__popc(ray & mask_above);
	}

}

// dodaje pola lądowania dla bicia damką przed rozgalezieniem albo finalne
__device__  void addKingLandingSquare(
	uint32_t occ_total,
	uint32_t occ_enemy,
	uint8_t capture_from,
	int8_t enemy_idx,
	uint8_t dir,
	uint32_t& move_packed,
	uint32_t& seed,
	uint8_t& count) {

	bool found_square = false;
	int8_t it = NEIGHBOURS[enemy_idx][dir];

	//printf("[top landin] from %d, with %d, it %d, dir %d\n", capture_from, enemy_idx, it, dir);
	while (it != -1) {
		//printf("it in while: %d\n", it);
		// zajete pole na glownej linii z mozliwoscia bicia
		if (occ_enemy & (1 << it)) {
			if (NEIGHBOURS[it][dir] != -1 && !(occ_total & (1 << NEIGHBOURS[it][dir]))) {
				//printf("Captuere in main line dir %d\n", dir);

				addKingBranchingCapture(move_packed, seed, count, capture_from, enemy_idx, NEIGHBOURS[it][(dir + 2) % 4], dir, it);
				found_square = true;
			}
		}

		// zajete pole bez mozliwosci bicia
		if (occ_total & (1 << it)) {
			break;
		}

		// mozliwosc rozgalezienia bicia
		int8_t next_enemy_idx = countEmptySquares(occ_total, occ_enemy, it, (dir + 1) % 4);
		//printf("Count %d dir %d\n", next_enemy_idx, (dir + 1) % 4);
		if (next_enemy_idx < 0) {
			//printf("Captuere dir %d\n", (dir + 1) % 4);

			addKingBranchingCapture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 1) % 4, -next_enemy_idx);
			found_square = true;
		}

		next_enemy_idx = countEmptySquares(occ_total, occ_enemy, it, (dir + 3) % 4);
		//printf("Count %d dir %d\n", next_enemy_idx, (dir + 3) % 4);
		if (next_enemy_idx < 0) {
			//printf("Captuere dir %d\n", (dir + 3) % 4);
			addKingBranchingCapture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 3) % 4, -next_enemy_idx);
			found_square = true;
		}


		it = NEIGHBOURS[it][dir];
	}

	//printf("\nfrom %d, with %d after_enemy %d dir %d found square %d\n", capture_from, enemy_idx, NEIGHBOURS[enemy_idx][dir], dir, found_square);
	if (!found_square) {
		//printf("Normal landing squares\n");
		it = NEIGHBOURS[enemy_idx][dir];
		while (it != -1 && !(occ_total & (1 << it))) {
			addKingFinalCapture(move_packed, seed, count, capture_from, enemy_idx, it);
			it = NEIGHBOURS[it][dir];
		}
	}

}


// obsluga ruchow damka
__device__  void chooseKingMove(
	uint32_t& move_packed,
	uint32_t& rng_reg,
	uint8_t& count,
	uint32_t occ_total,
	uint32_t occ_enemy,
	bool& was_capture,

	uint8_t index
) {
	int8_t ret[4];
	uint8_t i;
	bool found_capture = false;

	// sprawdz czy w jakims kierunku jest bicie
	for (i = 0; i < 4; i++) {
		ret[i] = countEmptySquares(occ_total, occ_enemy, index, i);
		//printf("ret: %d\n", ret[i]);
		if (ret[i] < 0) {
			found_capture = true;
		}
	}

	// znalezoino bicie usuniecie wszystkich normalnych ruchow
	if (!was_capture && found_capture) {
		move_packed = 0;
		count = 0;
		was_capture = true;
	}

	// nie znaleziono bicia a bylo juz bicie wczesniej
	//printf("Was cature: %d\n", was_capture);
	if (was_capture && !found_capture) {
		return;
	}

	if (found_capture) {
		//printf("found capture: %d\n", found_capture);
		// dodanie bicia i pol na ktorych moze wyladowac pionek przed rozgalezieniem
		for (i = 0; i < 4; i++) {
			if (ret[i] < 0) {
				addKingLandingSquare(
					occ_total,
					occ_enemy,
					index,
					-ret[i],
					i,
					move_packed,
					rng_reg,
					count
				);
			}
		}
	}
	else {
		// dodanie normalnych ruchow
		for (i = 0; i < 4; i++) {
			int8_t current = index;
			for (uint8_t j = 0; j < ret[i]; j++) {
				current = NEIGHBOURS[current][i];
				addNormalMove(
					move_packed,
					rng_reg,
					count,
					index,
					current
				);
			}

		}
	}
}

// wybiera losowy ruch z mozliwych
__device__  uint32_t chooseMove(
	uint32_t player_pawns,
	uint32_t player_kings,
	uint32_t occ_opponenet,
	uint32_t occ_total,
	bool is_white_move,
	uint32_t& seed
) {
	uint8_t counter = 0;
	bool is_capture = false;

	uint32_t move_packed = 0;
	int8_t with, to;


	// obsluga damek
	uint32_t figures = player_kings;
	while (figures)
	{
		uint8_t i = __ffs(figures) - 1;
		figures &= ~(1u << i);

		//printf("Checking king moves for index %d\n", i);

		chooseKingMove(
			move_packed,
			seed,
			counter,
			occ_total,
			occ_opponenet,
			is_capture,
			i
		);
	}

	// obsluga zwyklych pionkow
	figures = player_pawns;
	while (figures)
	{
		// ffs - index pierwszego bitu 1 od najmniej znaczacego
		uint8_t i = __ffs(figures) - 1;
		figures &= ~(1u << i);


		// bicie pionek
		for (char j = 0; j < 4; j++) {
			with = NEIGHBOURS[i][j];
			to = CAPTURES[i][j];
			if (with == -1 || to == -1) {
				continue;
			}

			if ((occ_opponenet & (1 << with))) {
				if (!(occ_total & (1 << to))) {
					if (is_capture == false) {
						counter = 0;

						is_capture = true;
					}
					addPawnCapture(
						move_packed,
						seed,
						counter,
						(uint8_t)i,
						(uint8_t)with,
						(uint8_t)to
					);
				}
			}
		}

		// normalny ruch
		if (is_capture == false) {
			if (is_white_move) {
				with = NEIGHBOURS[i][0];
			}
			else {
				with = NEIGHBOURS[i][2];
			}
			if (with != -1) {
				if (!(occ_total & (1 << with))) {
					addNormalMove(
						move_packed,
						seed,
						counter,
						(uint8_t)i,
						(uint8_t)with
					);
				}
			}
			if (is_white_move) {
				with = NEIGHBOURS[i][1];
			}
			else {
				with = NEIGHBOURS[i][3];
			}

			if (with != -1) {
				if (!(occ_total & (1 << with))) {
					addNormalMove(
						move_packed,
						seed,
						counter,
						(uint8_t)i,
						(uint8_t)with
					);
				}
			}
		}

	}


	//printf("Board counter: %d\n", counter);
	return move_packed;
}





__device__  void performNormalMove(
	uint32_t& player_pawns,
	uint32_t& player_kings,
	int8_t from,
	int8_t to
) {
	//printf("Normal move\n");
	if (player_pawns & (1 << from)) {
		player_pawns ^= (1 << from);
		if (to % 4 == 3 && ((to / 4) % 2) == 0 ||
			to % 4 == 0 && ((to / 4) % 2) == 1) {
			player_kings ^= (1 << to);

		}
		else {
			player_pawns ^= (1 << to);
		}

	}
	else {
		player_kings ^= (1 << from);
		player_kings |= (1 << to);
	}
}


__device__  void performPawnCapture(
	uint32_t& seed,
	uint32_t& player_pawns,
	uint32_t& player_kings,
	uint32_t& opponent_pawns,
	uint32_t& opponent_kings,
	int8_t from,
	int8_t with,
	int8_t to,
	bool is_white_move
) {
	//printf("Capture %d %d %d\n", from, with, to);

	player_pawns ^= 1 << from;
	uint8_t final_to;
	from = to;
	while (1) {
		//printf("while\n");
		uint32_t next_move = 0;
		uint8_t count = 0;
		from = to;
		final_to = to;

		opponent_pawns &= ~(1 << with);
		opponent_kings &= ~(1 << with);
		//printf("After\n");


		for (char j = 0; j < 4; j++) {
			with = NEIGHBOURS[from][j];
			to = CAPTURES[from][j];
			if (with == -1 || to == -1) {
				continue;
			}

			if (((opponent_pawns | opponent_kings) & (1 << with))) {
				if (!((opponent_kings | opponent_pawns | player_pawns | player_kings) & (1 << to))) {
					addPawnCapture(
						next_move,
						seed,
						count,
						(uint8_t)from,
						(uint8_t)with,
						(uint8_t)to
					);


				}
			}

		}
		if (next_move == 0) {
			break;
		}
		uint8_t placeholder;
		unpackMove(
			next_move,
			(uint8_t&)from,
			(uint8_t&)with,
			(uint8_t&)to,
			placeholder,
			placeholder,
			placeholder
		);
		/*
		printf("Unpacked next move: %d %d %d\n",
			(uint8_t&)from,
			(uint8_t&)with,
			(uint8_t&)to);*/


	}

	//printf("Final to %d\n", final_to);
	if ((is_white_move && final_to % 4 == 3 && ((final_to / 4) % 2) == 0) ||
		(!is_white_move && final_to % 4 == 0 && ((final_to / 4) % 2) == 1)) {
		player_kings |= 1 << final_to;
	}
	else
		player_pawns |= 1 << final_to;
}


__device__  void perfromKingCapture(
	uint32_t& seed,
	uint32_t& player_pawns,
	uint32_t& player_kings,
	uint32_t& opponent_pawns,
	uint32_t& opponent_kings,
	int8_t from,
	int8_t with,
	int8_t to,
	uint8_t type,
	uint8_t dir,
	int8_t next) {
	/*printf("King capture %d %d %d dir %d next %d\n", from, with, to, dir, next);
	printf("type %d\n", type);*/

	player_kings &= ~(1 << from);
	while (1) {
		//printf("King capture loop\n");
		opponent_kings &= ~(1 << with);
		opponent_pawns &= ~(1 << with);
		if (type == MOVE_TYPE_KING_FINAL_CAPTURE) {
			//printf("King final capture\n");

			player_kings |= (1 << to);

			break;
		}
		if (type == MOVE_TYPE_KING_BRANCHING_CAPTURE) {
			//printf("King branching capture\n");
			uint32_t next_move = 0;
			uint8_t count = 0;

			addKingLandingSquare(
				opponent_kings | opponent_pawns | player_kings | player_pawns,
				opponent_kings | opponent_pawns,
				to,
				next,
				dir,
				next_move,
				seed,
				count);

			unpackMove(
				next_move,
				(uint8_t&)from,
				(uint8_t&)with,
				(uint8_t&)to,
				(uint8_t&)type,
				(uint8_t&)dir,
				(uint8_t&)next
			);
			//printf("Next move packed: %u\n %d %d %d dir %d type %d next %d\n", next_move, from, with, to, dir, type, next);
		}
	}

}


__device__ void performMove(
	uint32_t& player_pawns,
	uint32_t& player_kings,
	uint32_t& opponent_pawns,
	uint32_t& opponent_kings,
	uint32_t move_packed,
	bool is_white_move,
	uint32_t& seed
) {

	uint8_t from, with, to, type, dir, next;
	unpackMove(
		move_packed,
		from,
		with,
		to,
		type,
		dir,
		next
	);
	//printf("\n\n\n\n Performing move \n\n\n");
	//printf("Move packed: %u\n %d %d %d dir %d next %d \n  type %d\n", move_packed, from, with, to, dir, next, type);

	switch (type) {
	case MOVE_TYPE_NORMAL:
		performNormalMove(player_pawns, player_kings, from, to);
		break;
	case MOVE_TYPE_PAWN_CAPTURE:
		performPawnCapture(seed, player_pawns, player_kings, opponent_pawns, opponent_kings, from, with, to, is_white_move);
		break;
	default:
		perfromKingCapture(seed, player_pawns, player_kings, opponent_pawns, opponent_kings, from, with, to, type, dir, next);
		break;
	}

}







#define NO_PROGRES_LIMIT 30 // po 15 ruchow z dwoch stron bez progresu gra jest przerywana (przegrywa ten ktory wykon aruch bez progresu)
#define MIN_MOVES 200 // po 100 ruchach z kazdej strony mozliwe jest przerwanie gry przy znaczenej przewadze (3 razy wiecej materialu)

// simulates a single game
// return true if white wins
__device__ bool simulate(
	uint32_t white_pawns,
	uint32_t white_kings,
	uint32_t black_pawns,
	uint32_t black_kings,
	uint32_t& seed
)
{
	
	uint32_t move_packed;
	bool is_white_move = true;
	bool is_white_winner = false;


	uint8_t no_progres_counter = 0;
	uint16_t total_moves = 0;


	uint8_t white_strength = __popc(white_pawns) + 2 * __popc(white_kings);
	uint8_t black_strength = __popc(black_pawns) + 2 * __popc(black_kings);

	while (1) {
		total_moves++;
		uint32_t occ_total = white_pawns | black_pawns | white_kings | black_kings;
		move_packed = 0;
		
		uint32_t old_white_pawns = white_pawns;
		uint32_t old_black_pawns = black_pawns;


		uint8_t old_white_strength = white_strength;
		uint8_t old_black_strength = black_strength;


		switch (is_white_move) {
		case true:
			//printf("Choosing white move\n");
			move_packed = chooseMove(white_pawns, white_kings, black_kings | black_pawns, occ_total, true, seed);
			performMove(white_pawns, white_kings, black_pawns, black_kings, move_packed, true, seed);

			break;
		case false:
			//printf("Choosing black move\n");
			move_packed = chooseMove(black_pawns, black_kings, white_kings | white_pawns, occ_total, false, seed);
			performMove(black_pawns, black_kings, white_pawns, white_kings, move_packed, false, seed);

			break;
		}
		white_strength = __popc(white_pawns) + 2 * __popc(white_kings);
		black_strength = __popc(black_pawns) + 2 * __popc(black_kings);



		/*printBoard(
			black_pawns,
			white_pawns,
			black_kings,
			white_kings);*/


		// brak progresu
		if(
			old_black_pawns == black_pawns &&
			old_white_pawns == white_pawns &&
			old_white_strength == white_strength &&
			old_black_strength == black_strength

			) {
			no_progres_counter++;
			if(no_progres_counter > NO_PROGRES_LIMIT) {
				//printf("BRAK PROGRESSU BREAKING\n");
				is_white_winner = !is_white_move;
				break;
			}
		}
		else {
			no_progres_counter = 0;
		}

		// uzyskanie pzewagi po pewnej liczbie ruchow
		if (total_moves > MIN_MOVES) {
			//printf("white strength %d black strength %d\n", white_strength, black_strength);
			if(white_strength >= black_strength * 3) {
				is_white_winner = true;
				//printf("PRZEWAGA BIALY BREAKING\n");
				break;
			}

			if (black_strength >= white_strength * 3) {
				is_white_winner = false;
				//printf("PRZEWAGA CZARNY BREAKING\n");
				break;
			}
		}


		// brak mozliwych ruchow
		if (move_packed == 0) {
			is_white_winner = !is_white_move;
			//printf("DIDINT FOUND MOVE BREAKING\n");
			break;
		}

		is_white_move = !is_white_move;
	}

	return is_white_winner;

}


__device__ __forceinline__ uint32_t makeSeed(uint32_t base, uint32_t tid) {
	uint32_t x = base ^ (tid * 0x9E3779B9u);  
	x ^= x >> 16;
	x *= 0x85EBCA6Bu;
	x ^= x >> 13;
	x *= 0xC2B2AE35u;
	x ^= x >> 16;
	return x | 1u;
}

__global__ void MCTSKernel(Board* b, char* ret) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t seed = makeSeed(1234567, tid);

	bool is_white_winner = simulate(
		b->white_pawns,
		b->white_kings,
		b->black_pawns,
		b->black_kings,
		seed
	);

	/*if (is_white_winner) {
		printf("White winner id: %d\n", tid);
	}
	else {
		printf("Black winner id %d\n", tid);
	}*/
	// to be implemented
}


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


cudaError_t calcAllMovesCuda(Board board_in) {
	cudaError_t cudaStatus = cudaSuccess;

	Board* d_in = nullptr;
	char* d_ret = nullptr;

	cudaEvent_t startEv = nullptr, stopEv = nullptr;
	float ms = 0.0f;

	// <-- PRZENIESIONE TU, żeby goto nie omijało inicjalizacji
	float avg_ms = 0.0f;
	double threads_total = 0.0;
	double threads_per_sec = 0.0;

	const int grid = 100000;
	const int block = 128;

	const int warmupIters = 3;
	const int measureIters = 5;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
		goto Error;
	}

	buildNeighbourTabs();
	cudaStatus = cudaMemcpyToSymbol(NEIGHBOURS, Neighbours, sizeof(Neighbours));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Memcpy NEIGHBOURS failed!\n"); goto Error; }

	cudaStatus = cudaMemcpyToSymbol(CAPTURES, Captures, sizeof(Captures));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Memcpy CAPTURES failed!\n"); goto Error; }

	buildRayTab();
	cudaStatus = cudaMemcpyToSymbol(RAYS, Rays, sizeof(Rays));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Memcpy RAYS failed!\n"); goto Error; }

	cudaStatus = cudaMalloc((void**)&d_in, sizeof(Board));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_in failed!\n"); goto Error; }

	cudaStatus = cudaMalloc((void**)&d_ret, 32 * sizeof(char));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_ret failed!\n"); goto Error; }

	cudaStatus = cudaMemcpy(d_in, &board_in, sizeof(Board), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_in failed!\n"); goto Error; }

	cudaStatus = cudaEventCreate(&startEv);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventCreate(start) failed!\n"); goto Error; }

	cudaStatus = cudaEventCreate(&stopEv);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventCreate(stop) failed!\n"); goto Error; }

	// warmup
	for (int i = 0; i < warmupIters; ++i) {
		MCTSKernel << <grid, block >> > (d_in, d_ret);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MCTSKernel warmup launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize after warmup failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// measured batch
	cudaStatus = cudaEventRecord(startEv, 0);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventRecord(start) failed!\n"); goto Error; }

	for (int i = 0; i < measureIters; ++i) {
		MCTSKernel << <grid, block >> > (d_in, d_ret);
	}

	cudaStatus = cudaEventRecord(stopEv, 0);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventRecord(stop) failed!\n"); goto Error; }

	cudaStatus = cudaEventSynchronize(stopEv);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventSynchronize(stop) failed!\n"); goto Error; }

	cudaStatus = cudaEventElapsedTime(&ms, startEv, stopEv);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventElapsedTime failed!\n"); goto Error; }

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MCTSKernel measured launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	avg_ms = ms / (float)measureIters;
	threads_total = (double)grid * (double)block;
	threads_per_sec = threads_total / (avg_ms * 1e-3);

	printf("MCTSKernel<<<%d,%d>>> total: %.3f ms for %d launches, avg: %.6f ms/launch\n",
		grid, block, ms, measureIters, avg_ms);
	printf("Throughput: %.3f M threads/s\n", threads_per_sec / 1e6);

Error:
	if (stopEv)  cudaEventDestroy(stopEv);
	if (startEv) cudaEventDestroy(startEv);
	if (d_ret)   cudaFree(d_ret);
	if (d_in)    cudaFree(d_in);

	return cudaStatus;
}


//int main()
//{
//
//	Board board = startBoard();
//	printBoard(board);
//
//	calcAllMovesCuda(board);
//
//	return 0;
//}


