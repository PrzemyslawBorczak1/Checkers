
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tests.h"




__device__ __constant__  int8_t NEIGHBOURS[32][4];
__device__ __constant__  int8_t CAPTURES[32][4];

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




__device__ int8_t countEmptySquares(
	uint32_t occ_total,
	uint32_t occ_enemy,
	uint8_t index,
	uint8_t dir
) {

	int8_t ind = NEIGHBOURS[index][dir];
	int8_t ret = 0;

	while (ind != -1) {
		// pole zajete
		if (occ_total & (1 << ind)) {
			// zajete przez przeciwnika
			if (occ_enemy & (1 << ind)) {
				printf("enemy  %d\n", ind);
				// mozliwe jest wybicie
				if (NEIGHBOURS[ind][dir] != -1 && !(occ_total & (1 << NEIGHBOURS[ind][dir]))) {
					printf("Captiure ended in  %d\n", NEIGHBOURS[ind][dir]);
					return -(ret + 2);
				}
				else {
					break;
				}
			}
			else {
				break;
			}
		}

		ind = NEIGHBOURS[ind][dir];
		ret++;
	}
	return ret;
}



__device__ void addLandingSquareCapture_new(
	uint32_t occ_total,
	uint32_t occ_enemy,
	uint8_t index,
	uint8_t dir,
	int8_t skip,
	uint32_t& move_packed,
	uint32_t& rng_reg,
	uint8_t& count) {

	uint8_t i = 0;
	int8_t ind = index;
	bool found_square = false;
	for (i = 0; i < skip - 1; i++) {
		ind = NEIGHBOURS[ind][dir];
	}
	int8_t start = ind;
	ind = NEIGHBOURS[ind][dir];

	printf("ind: %d fro idx: %d init skip  %d\n", ind, index, skip);
	while (ind != -1){
		if (occ_total & (1 << ind)) {
			break;
		}

		if (countEmptySquares(occ_total, occ_enemy, ind, (dir + 1) % 4) < 0) {
			printf("Captuere dir %d\n", (dir + 1) % 4);
			add_new_dir_capture(move_packed, rng_reg, count, index, start, ind, (dir + 1) % 4);
			found_square = true;
		}
		if (countEmptySquares(occ_total, occ_enemy, ind, (dir + 3) % 4) < 0) {
			printf("Captuere dir %d\n", (dir + 3) % 4);
			add_new_dir_capture(move_packed, rng_reg, count, index, start, ind, (dir + 3) % 4);
			found_square = true;
		}

		if (NEIGHBOURS[ind][dir] != -1 && occ_enemy & (1 << NEIGHBOURS[ind][dir])) {
			if (CAPTURES[ind][dir] != -1 && !(occ_total & (1 << CAPTURES[ind][dir]))) {
				printf("Captuere in old line dir %d\n", (dir + 3) % 4);
				add_new_dir_capture(move_packed, rng_reg, count, index, start, ind, dir);
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
			add_new_dir_capture(move_packed, rng_reg, count, index, start, ind, NO_DIRECTION);
			ind = NEIGHBOURS[ind][dir];
			
		}
	}

}







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

	for (i = 0; i < 4; i++) {
		ret[i] = countEmptySquares(occ_total, occ_enemy, index, i);
		printf("ret: %d\n", ret[i]);
		if (ret[i] < 0) {
			found_capture = true;
		}
	}

	if (!was_capture && found_capture) {
		move_packed = 0;
		count = 0;
		was_capture = true;
	}

	printf("Was cature: %d\n", was_capture);
	if (was_capture && !found_capture) {
		return;
	}
	
	if (found_capture) {

		printf("found capture: %d\n", found_capture);
		for (i = 0; i < 4; i++) {
			if (ret[i] < 0) {
				addLandingSquareCapture_new(
					occ_total,
					occ_enemy,
					index,
					i,
					-ret[i],
					move_packed,
					rng_reg,
					count
					);
			}
		}
	}
	else{
		for (i = 0; i < 4; i++) {

			uint8_t j = 0;
			int8_t current = index;
			for (j = 0; j < ret[i]; j++) {
				current = NEIGHBOURS[current][i];
				add_new_normal(move_packed, rng_reg, count, index, current);
			}

		}
	}
}







__device__ uint32_t chooseMove(
	uint32_t player_pawns,
	uint32_t player_kings,
	uint32_t opponenet_occupied,
	uint32_t occupied_total,
	bool is_white_move,
	uint32_t& seed
) {
	uint8_t counter = 0;
	bool is_capture = false;

	uint32_t move_packed = 0;


	int8_t with, to;
	char board_index = 0;
	for (uint8_t i = 0; i < 32; i++) {
		
		if (!((player_pawns | player_kings) & (1 << i))) {
			continue;
		}

		if (player_pawns & 1 << i) {


			// bicie pionek
			for (char j = 0; j < 4; j++) {
				with = NEIGHBOURS[i][j];
				to = CAPTURES[i][j];
				if (with == -1 || to == -1) {
					continue;
				}

				if ((opponenet_occupied & (1 << with))) {
					if (!(occupied_total & (1 << to))) {
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

			// normal move
			if (is_capture == false) {
				if (is_white_move) {
					with = NEIGHBOURS[i][0];
				}
				else {
					with = NEIGHBOURS[i][2];
				}
				if (with != -1) {
					if (!(occupied_total & (1 << with))) {
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
					if (!(occupied_total & (1 << with))) {
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

		if (player_kings & 1 << i) {
			printf("Checking king moves for index %d\n", i);

			king_move_new(
				move_packed,
				seed,
				counter,
				occupied_total,
				opponenet_occupied,
				is_capture,
				i
			);
		}

		board_index++;
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
	for (int i = 0; i < 50; i++) {
		uint32_t occ_total = white_pawns | black_pawns | white_kings | black_kings;
		move_packed = 0;
		switch (move) {
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




int8_t Neighbours[32][4];
int8_t Captures[32][4];

void buildHelperTabs() {
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

cudaError_t calcAllMovesCuda(Board board_in) {


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	buildHelperTabs();
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

	Board board = capturePromotion();
	printBoard(board);
    
	calcAllMovesCuda(board);

    return 0;
}


