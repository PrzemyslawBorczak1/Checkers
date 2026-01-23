
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tests.h"




__device__ __constant__  int8_t NEIGHBOURS[32][4];
__device__ __constant__  int8_t CAPTURES[32][4];
__device__ __constant__  uint32_t RAYS[32][4];


int8_t Neighbours[32][4];
int8_t Captures[32][4];
uint32_t Rays[32][4];

// 1 - 4  damka nie skonczone bicie
#define NO_DIRECTION 5 // damka skonczone bicie
#define UNKOWN_DIRECTION 6 // pionek bicie


__device__ __forceinline__ void unpack_u16_move(
	uint32_t packed,
	uint8_t& from,
	uint8_t& with,
	uint8_t& to,
	uint8_t& dir
) {
	from = (uint8_t)(packed & 31u);
	with = (uint8_t)((packed >> 5) & 31u);
	to = (uint8_t)((packed >> 10) & 31u);
	dir = (uint8_t)((packed >> 15) & 31u);
}


__device__ __forceinline__ void add_new_capture(
	uint32_t& move_pacekd,
	uint32_t& rng_reg,
	uint8_t& count,
	uint8_t from,
	uint8_t with,
	uint8_t to
) {
	printf("Add new pawn capture %d %d %d  count %d\n", from, with, to, count);
	count++;
	uint32_t x = rng_reg;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_reg = x;
	if ((x % (uint32_t)count) == 0u) {
		move_pacekd = (uint32_t)0 | (((to & 31u) << 10) |
			((with & 31u) << 5) |
			((from & 31u)));
	}
}

__device__ __forceinline__ void add_new_normal(
	uint32_t& move_packed,
	uint32_t& rng_reg,
	uint8_t& count,
	uint8_t from,
	uint8_t to
) {
	printf("Add new normal move %d %d  count %d\n", from, to, count);
	count++;
	uint32_t x = rng_reg;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_reg = x;
	if ((x % (uint32_t)count) == 0u) {
		move_packed = (uint32_t)0 | (((to & 31u) << 10) |
			((from & 31u)));
	}
}

__device__ __forceinline__ void add_new_dir_capture(
	uint32_t& move_pacekd,
	uint32_t& rng_reg,
	uint8_t& count,
	uint8_t from,
	uint8_t with,
	uint8_t to,
	uint8_t dir
) {
	printf("Add new king capture %d %d %d dir %d count %d\n", from, with, to,dir,  count);
	count++;
	uint32_t x = rng_reg;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_reg = x;
	if ((x % (uint32_t)count) == 0u) {
		move_pacekd = 
			((uint32_t)(dir & 31u) << 15) |
			((uint32_t)(to & 31u) << 10) |
			((uint32_t)(with & 31u) << 5) |
			((uint32_t)(from & 31u));
	}
}








// zwraca liczbe wolnych pol w danym kierunku
// lub -index przeciwnika w przypadku bicia (nigdy 0)
__device__ int8_t countEmptySquares(
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
__device__ void addLandingSquareCapture_new(
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

	printf("[top landin] from %d, with %d, it %d, dir %d\n", capture_from, enemy_idx, it, dir);
	while (it != -1){
		printf("it in while: %d\n", it);
		// zajete pole na glownej linii z mozliwoscia bicia
		if (occ_enemy & (1 << it)) {
			if (NEIGHBOURS[it][dir] != -1 && !(occ_total & (1 << NEIGHBOURS[it][dir]))) {
				printf("Captuere in main line dir %d\n", dir);
				add_new_dir_capture(move_packed, seed, count, capture_from, enemy_idx, NEIGHBOURS[it][dir], dir);
				found_square = true;
			}
		}

		// zajete pole bez mozliwosci bicia
		if (occ_total & (1 << it)) {
			break;
		}

		// mozliwosc rozgalezienia bicia
		if (countEmptySquares(occ_total, occ_enemy, it, (dir + 1) % 4) < 0) {
			printf("Captuere dir %d\n", (dir + 1) % 4);
			add_new_dir_capture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 1) % 4);
			found_square = true;
		}
		if (countEmptySquares(occ_total, occ_enemy, it, (dir + 3) % 4) < 0) {
			printf("Captuere dir %d\n", (dir + 3) % 4);
			add_new_dir_capture(move_packed, seed, count, capture_from, enemy_idx, it, (dir + 3) % 4);
			found_square = true;
		}


		it = NEIGHBOURS[it][dir];
	}

	printf("\nfrom %d, with %d after_enemy %d dir %d found square %d\n", capture_from, enemy_idx, NEIGHBOURS[enemy_idx][dir], dir, found_square);
	if (!found_square) {
		printf("Normal landing squares\n");
		it = NEIGHBOURS[enemy_idx][dir];
		while (it != -1 && !(occ_total & (1 << it))) {
			add_new_dir_capture(move_packed, seed, count, capture_from, enemy_idx, it, NO_DIRECTION);
			it = NEIGHBOURS[it][dir];			
		}
	}

}


// obsluga ruchow damka
__device__ __forceinline__ void king_move_new(
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
		printf("ret: %d\n", ret[i]);
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
	printf("Was cature: %d\n", was_capture);
	if (was_capture && !found_capture) {
		return;
	}
	
	if (found_capture) {
		printf("found capture: %d\n", found_capture);
		// dodanie bicia i pol na ktorych moze wyladowac pionek przed rozgalezieniem
		for (i = 0; i < 4; i++) {
			if (ret[i] < 0) {
				addLandingSquareCapture_new(
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
	else{
		// dodanie normalnych ruchow
		for (i = 0; i < 4; i++) {
			int8_t current = index;
			for (uint8_t j = 0; j < ret[i]; j++) {
				current = NEIGHBOURS[current][i];
				add_new_normal(move_packed, rng_reg, count, index, current);
			}

		}
	}
}


__device__ uint32_t chooseMove(
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

		printf("Checking king moves for index %d\n", i);

		king_move_new(
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
					add_new_dir_capture(
						move_packed,
						seed,
						counter,
						(uint8_t)i,
						(uint8_t)with,
						(uint8_t)to,
						UNKOWN_DIRECTION
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
					add_new_normal(
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
					add_new_normal(
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
	

	printf("Board counter: %d\n", counter);
	return move_packed;
}








__device__ uint32_t chooseKingNextMove(
	uint32_t occ_total,
	uint32_t occ_enemy,
	uint8_t from,
	uint8_t dir,
	uint32_t& move_packed,
	uint32_t& rng_reg,
	uint8_t& count){


		uint8_t i = 0;
		int8_t ind = from;
		bool found_square = false;
		while(!(occ_total & (1 << ind))){
			printf("ind whiule 1 init: %d\n", ind);
			ind = NEIGHBOURS[ind][dir];
			if(ind == -1){
				break;
			}
		}
		printf("Next move ind: %d\n", ind);

		int8_t start = ind;
		ind = NEIGHBOURS[ind][dir];

		while (ind != -1) {
			printf("ind while 2 init: %d\n", ind);
			if (occ_total & (1 << ind)) {
				break;
			}

			if (countEmptySquares(occ_total, occ_enemy, ind, (dir + 1) % 4) < 0) {
				printf("Captuere dir %d\n", (dir + 1) % 4);
				add_new_dir_capture(move_packed, rng_reg, count, from, start, ind, (dir + 1) % 4);
				found_square = true;
			}
			if (countEmptySquares(occ_total, occ_enemy, ind, (dir + 3) % 4) < 0) {
				printf("Captuere dir %d\n", (dir + 3) % 4);
				add_new_dir_capture(move_packed, rng_reg, count, from, start, ind, (dir + 3) % 4);
				found_square = true;
			}

			if (NEIGHBOURS[ind][dir] != -1 && occ_enemy & (1 << NEIGHBOURS[ind][dir])) {
				if (CAPTURES[ind][dir] != -1 && !(occ_total & (1 << CAPTURES[ind][dir]))) {
					add_new_dir_capture(move_packed, rng_reg, count, from, start, ind, dir);
					found_square = true;
				}
			}


			ind = NEIGHBOURS[ind][dir];
		}

		if (!found_square) {
			ind = NEIGHBOURS[start][dir];
			while (ind != -1) {
				if (occ_total & (1 << ind)) {
					break;
				}
				add_new_dir_capture(move_packed, rng_reg, count, from, start, ind, NO_DIRECTION);
				ind = NEIGHBOURS[ind][dir];

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

	uint8_t from, with, to, dir;
	unpack_u16_move(move_packed, from, with, to, dir);
	printf("\n\n\n\n PErforming move \n\n\n");
	printf("Move packed: %u\n %d %d %d dir %d\n", move_packed, from, with, to, dir);

	bool perfomred = false;
	// normal move
	if (with == 0) {
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
		perfomred = true;
	}
	// znicie pionek
	if (dir == UNKOWN_DIRECTION && !perfomred) {
		printf("Capture\n");

		int8_t new_from, new_with, new_to, new_dir;

		player_pawns ^= 1 << from;

		 new_from = to;
		while (dir == UNKOWN_DIRECTION) {
			printf("while\n");
			uint32_t next_move = 0;
			uint8_t count = 0;
			new_from = to;

			opponent_pawns &= ~(1 << with);
			opponent_kings &= ~(1 << with);
			/*printf("\nColor may not be right:\n");
			printBoard(
				player_pawns,
				opponent_pawns,
				player_kings,
				opponent_kings
			);*/

			for (char j = 0; j < 4; j++) {
				printf("fro\n");
				new_with = NEIGHBOURS[new_from][j];
				new_to = CAPTURES[new_from][j];
				if (new_with == -1 || new_to == -1) {
					continue;
				}

				if (((opponent_pawns | opponent_kings) & (1 << new_with))) {
					if (!((opponent_kings | opponent_pawns | player_pawns | player_kings) & (1 << new_to))) {

						add_new_dir_capture(
							next_move,
							seed,
							count,
							(uint8_t)new_from,
							(uint8_t)new_with,
							(uint8_t)new_to,
							UNKOWN_DIRECTION
						);
					}
				}

			}
			unpack_u16_move(next_move, from, with, to, dir);
			printf("Unpacked next move: %d %d %d dir %d\n", from, with, to, dir);
		}

		printf("Final to %d\n", new_from);
		if((is_white_move && new_from % 4 == 3 && ((new_from / 4) % 2) == 0) || 
			(!is_white_move && new_from % 4 == 0 && ((new_from / 4) % 2) == 1)) {
			player_kings |= 1 << new_from;
		}
		else
		player_pawns |= 1 << new_from;
		perfomred = true;
	}



	while (dir <= NO_DIRECTION && !perfomred) {
		printf("King capture\n");
		// king single capture
		if (dir == NO_DIRECTION) {
			printf("King single capture\n");
			opponent_kings &= ~(1 << with);
			opponent_pawns &= ~(1 << with);

			player_kings &= ~(1 << from);
			player_kings |= (1 << to);

			perfomred = true;
			break;
		}
		// king multiple capture
		if (dir < NO_DIRECTION) {
			printf("King multipe capture\n");
			opponent_kings &= ~(1 << with);
			opponent_pawns &= ~(1 << with);

			player_kings &= ~(1 << from);

			uint32_t next_move = 0;
			uint8_t count = 0;


			chooseKingNextMove(
				opponent_kings | opponent_pawns | player_kings | player_pawns,
				opponent_kings | opponent_pawns,
				to,
				dir,
				next_move,
				seed,
				count);

			unpack_u16_move(next_move, from, with, to, dir);
			printf("Next move packed: %u\n %d %d %d dir %d\n", next_move, from, with, to, dir);
		}

		
	}
}










__global__ void checkersKernel(Board* b, char* ret)
{
	// todo add sth like thsi
	/*int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t seed = base_seed ^ (uint32_t)tid;
	seed = seed ? seed : 1u;*/
	uint32_t seed = 12123u;


	uint32_t white_pawns = b->white_pawns;
	uint32_t black_pawns = b->black_pawns;

	uint32_t black_kings = b->black_kings;
	uint32_t white_kings = b->white_kings;



	int thread_index = threadIdx.x;
	uint32_t move_packed;

	bool move = true;
	for (int i = 0; i < 1; i++) {
		uint32_t occ_total = white_pawns | black_pawns | white_kings | black_kings;
		move_packed = 0;
		switch (true) {
		case true:
			printf("Choosing white move\n");
			move_packed = chooseMove(white_pawns, white_kings, black_kings | black_pawns, occ_total, true, seed);
			performMove(white_pawns, white_kings, black_pawns, black_kings, move_packed, true, seed);
			


			//uint8_t from, with, to, dir;
			//unpack_u16_move(move_packed, from, with, to, dir);
			//printf("Move packed: %u\n %d %d %d dir %d\n", move_packed, from, with, to, dir);

			break;

		case false:
			printf("Choosing black move\n");
			move_packed = chooseMove(black_pawns, black_kings, white_kings | white_pawns, occ_total, false, seed);
			performMove(black_pawns, black_kings, white_pawns, white_kings, move_packed, false, seed);
			//printBoard(b[0]);
			// from, with, to, dir;
			//unpack_u16_move(move_packed, from, with, to, dir);
			//printf("Move packed: %u\n %d %d %d dir %d\n", move_packed, from, with, to, dir);

			break;
		}
		if (move_packed == 0) {
			printf("DIDINT FOUND MOVE BREAKING\n");
			break;
		}
		printBoard(
			black_pawns,
			white_pawns,
			black_kings,
			white_kings);
		move = !move;
	}


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


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	buildNeighbourTabs();
	cudaStatus = cudaMemcpyToSymbol(NEIGHBOURS, Neighbours, sizeof(Neighbours));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(CAPTURES, Captures, sizeof(Captures));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	buildRayTab();
	cudaStatus = cudaMemcpyToSymbol(RAYS, Rays, sizeof(Rays));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}


	Board* d_in = nullptr;
	char d_ret[32];

	
	cudaStatus = cudaMalloc((void**)&d_in, sizeof(Board));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_ret, 32 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_in, &board_in, sizeof(Board), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	checkersKernel << <1, 1 >> > (d_in, d_ret);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkersKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching checkersKernel!\n", cudaStatus);
		goto Error;
	}

	

	return cudaSuccess;

Error:

	return cudaStatus;
}



int main()
{	

	Board board = kingAllDir();
	printBoard(board);
    
	calcAllMovesCuda(board);

    return 0;
}


