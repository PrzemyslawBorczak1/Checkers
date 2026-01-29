#pragma once
#include "mcts_kernel.cuh"



__device__ __constant__  int8_t NEIGHBOURS[32][4];
__device__ __constant__  int8_t CAPTURES[32][4];
__device__ __constant__  uint32_t RAYS[32][4];


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

	while (it != -1) {
		// zajete pole na glownej linii z mozliwoscia bicia
		if (occ_enemy & (1 << it)) {
			if (NEIGHBOURS[it][dir] != -1 && !(occ_total & (1 << NEIGHBOURS[it][dir]))) {

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
		if (next_enemy_idx < 0) {

			addKingBranchingCapture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 1) % 4, -next_enemy_idx);
			found_square = true;
		}

		next_enemy_idx = countEmptySquares(occ_total, occ_enemy, it, (dir + 3) % 4);
		if (next_enemy_idx < 0) {
			addKingBranchingCapture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 3) % 4, -next_enemy_idx);
			found_square = true;
		}


		it = NEIGHBOURS[it][dir];
	}

	if (!found_square) {
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
	if (was_capture && !found_capture) {
		return;
	}

	if (found_capture) {
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

	return move_packed;
}





__device__  void performNormalMove(
	uint32_t& player_pawns,
	uint32_t& player_kings,
	int8_t from,
	int8_t to
) {
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
	player_pawns ^= 1 << from;
	uint8_t final_to;
	from = to;
	while (1) {
		uint32_t next_move = 0;
		uint8_t count = 0;
		from = to;
		final_to = to;

		opponent_pawns &= ~(1 << with);
		opponent_kings &= ~(1 << with);

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

	}


	if ((is_white_move && (final_to % 4) == 3 && ((final_to / 4) % 2) == 0) ||
		(!is_white_move && (final_to % 4) == 0 && ((final_to / 4) % 2) == 1)) {
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
	player_kings &= ~(1 << from);
	while (1) {
		opponent_kings &= ~(1 << with);
		opponent_pawns &= ~(1 << with);
		if (type == MOVE_TYPE_KING_FINAL_CAPTURE) {
			player_kings |= (1 << to);

			break;
		}
		if (type == MOVE_TYPE_KING_BRANCHING_CAPTURE) {
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



// simulates a single game
// return true if white wins
__device__ bool simulate(
	uint32_t white_pawns,
	uint32_t white_kings,
	uint32_t black_pawns,
	uint32_t black_kings,
	uint32_t& seed,
	bool is_white_move,
	int moves_without_progres
)
{
	uint32_t move_packed;
	bool is_white_winner = false;


	uint8_t no_progres_counter = moves_without_progres;
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
			move_packed = chooseMove(white_pawns, white_kings, black_kings | black_pawns, occ_total, true, seed);
			performMove(white_pawns, white_kings, black_pawns, black_kings, move_packed, true, seed);

			break;
		case false:
			move_packed = chooseMove(black_pawns, black_kings, white_kings | white_pawns, occ_total, false, seed);
			performMove(black_pawns, black_kings, white_pawns, white_kings, move_packed, false, seed);

			break;
		}
		white_strength = __popc(white_pawns) + 2 * __popc(white_kings);
		black_strength = __popc(black_pawns) + 2 * __popc(black_kings);


		// brak progresu
		if(
			old_black_pawns == black_pawns &&
			old_white_pawns == white_pawns &&
			old_white_strength == white_strength &&
			old_black_strength == black_strength

			) {
			no_progres_counter++;
			if(no_progres_counter > NO_PROGRES_LIMIT) {
				if(white_strength > black_strength) {
					is_white_winner = true;
				}
				else if (black_strength > white_strength) {
					is_white_winner = false;
				}
				else {
					is_white_winner = !is_white_move;
				}

				break;
			}
		}
		else {
			no_progres_counter = 0;
		}

		// uzyskanie pzewagi po pewnej liczbie ruchow
		if (total_moves > MIN_MOVES) {
			if(white_strength >= black_strength * 3) {
				is_white_winner = true;
				break;
			}

			if (black_strength >= white_strength * 3) {
				is_white_winner = false;
				break;
			}
		}


		// brak mozliwych ruchow
		if (move_packed == 0) {
			is_white_winner = !is_white_move;
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


__device__ void reduce(int16_t* to_sum, uint16_t size, uint32_t* save) {
	for (uint16_t stride = (size + 1) >> 1; stride > 0; stride = (stride + 1) >> 1) {
		uint16_t i = (uint16_t)threadIdx.x;

		if (i < stride) {
			uint16_t j = i + stride;
			if (j < size) {
				uint16_t a = to_sum[i];
				uint16_t b = to_sum[j];
				uint16_t s = a + b;

				to_sum[i] = s;
			}
		}
		__syncthreads();

		size = stride;
		if (size == 1) break;
	}

	if (threadIdx.x == 0) {
		save[blockIdx.x] = to_sum[0];
	}
}

__global__ void mctsKernel(Board* b, uint32_t* ret, bool is_white_move, uint32_t seed_cpu, int moves_without_progres) {
	__shared__ int16_t winner[THREADS];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t seed = makeSeed(seed_cpu,tid);

	bool is_white_winner = simulate(
		b->white_pawns,
		b->white_kings,
		b->black_pawns,
		b->black_kings,
		seed,
		is_white_move,
		moves_without_progres
	);
	
	if (is_white_winner) {
		winner[threadIdx.x] = 1;
	}
	else {
		winner[threadIdx.x] = 0;
	}
	__syncthreads();
	reduce(winner, THREADS, ret);
}

__global__ void reduceKernel(uint32_t* to_sum, uint16_t size) {
	for (uint32_t stride = (size + 1) >> 1; stride > 0; stride = (stride + 1) >> 1) {
		uint32_t i = (uint32_t)threadIdx.x;

		if (i < stride) {
			uint32_t j = i + stride;
			if (j < size) {
				uint32_t a = to_sum[i];
				uint32_t b = to_sum[j];
				uint32_t s = a + b;

				to_sum[i] = s;
			}
		}
		__syncthreads();

		size = stride;
		if (size == 1) break;
	}
}

cudaError_t mctsSetSymbols(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]) {
	cudaError_t st;

	st = cudaSetDevice(0);
	if (st != cudaSuccess) return st;

	st = cudaMemcpyToSymbol(NEIGHBOURS, Neighbours, sizeof(Neighbours));
	if (st != cudaSuccess) return st;
	st = cudaMemcpyToSymbol(CAPTURES, Captures, sizeof(Captures));
	if (st != cudaSuccess) return st;

	st = cudaMemcpyToSymbol(RAYS, Rays, sizeof(Rays));
	if (st != cudaSuccess) return st;

	return cudaSuccess;

}


uint32_t runMCTS(Board* dev_board, uint32_t* dev_ret,  Color color, uint32_t seed, int moves_without_progres) {

	cudaError_t cs;
	bool is_white_move = (color == Color::WHITE);
	mctsKernel<<<BLOCKS,THREADS>>>(dev_board, dev_ret, is_white_move, seed, moves_without_progres);

	cs = cudaDeviceSynchronize();
	if (cs != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize  failed: %s\n", cudaGetErrorString(cs));
	}

	reduceKernel << <1, BLOCKS >> > (dev_ret, BLOCKS);

	cs = cudaDeviceSynchronize();
	if (cs != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize  failed: %s\n", cudaGetErrorString(cs));
	}


	uint32_t ret;
	cs = cudaMemcpy(&ret, dev_ret, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (cs != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cs));
	}


	return ret;
}


