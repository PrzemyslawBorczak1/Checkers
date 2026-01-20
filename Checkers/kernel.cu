
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tests.h"




__device__ __constant__  int8_t NEIGHBOURS[32][4];
__device__ __constant__  int8_t CAPTURES[32][4];



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

__device__ __forceinline__ void add_new_king_capture(
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
			add_new_king_capture(move_packed, rng_reg, count, index, start, ind, (dir + 1) % 4);
			found_square = true;
		}
		if (countEmptySquares(occ_total, occ_enemy, ind, (dir + 3) % 4) < 0) {
			printf("Captuere dir %d\n", (dir + 3) % 4);
			add_new_king_capture(move_packed, rng_reg, count, index, start, ind, (dir + 3) % 4);
			found_square = true;
		}

		if (occ_enemy & (1 << NEIGHBOURS[ind][dir])) {
			if (!(occ_total & (1 << CAPTURES[ind][dir]))) {
				add_new_king_capture(move_packed, rng_reg, count, index, start, ind, dir);
				found_square = true;
			}
		}


		ind = NEIGHBOURS[ind][dir];
	}

	if (!found_square) {
		ind = NEIGHBOURS[start][dir];
		while (ind != -1) {

			add_new_king_capture(move_packed, rng_reg, count, index, start, ind, 5);
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





















__global__ void checkersKernel(Board* b, char* ret)
{
	// todo add sth like thsi
	/*int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t seed = base_seed ^ (uint32_t)tid;
	seed = seed ? seed : 1u;*/
	uint32_t seed = 123456789u;


	uint32_t white_pawns = b->white_pawns;
	uint32_t black_pawns = b->black_pawns;

	uint32_t black_kings = b->black_kings;
	uint32_t white_kings = b->white_kings;


	uint32_t occupied_total = b->occupied_total;

	int thread_index = threadIdx.x;


	uint8_t counter;
	uint32_t move_packed;

	bool is_capture = false;

	char offset = 0;

	uint32_t opponenet_occupied;
	uint32_t player_pawns;
	uint32_t player_kings;

	switch (b->is_white_move) {
	case true:
		opponenet_occupied = black_pawns | black_kings;
		player_pawns = white_pawns;
		player_kings = white_kings;
		break;

	case false:
		opponenet_occupied = white_pawns | white_kings;
		player_pawns = black_pawns;
		player_kings = black_kings;
		break;
	}


	counter = 0;

	uint8_t with, to;
	char board_index = 0;

	for (uint8_t i = 0; i < 32; i++) {
		if (i % 4 == 0) {
			offset = 1 - offset;
		}

		if (!((player_pawns | player_kings) & (1 << i))) {
			continue;
		}

		if (player_pawns & 1 << i) {
			for (char j = 0; j < 4; j++) {
				with = NEIGHBOURS[i][j];
				to = CAPTURES[i][j];

				if (to < 0 || to >= 32) {
					continue;
				}

				if (i / 4 % 2 != to / 4 % 2) {
					continue;
				}

				if ((opponenet_occupied & (1 << with))) {
					if (!(occupied_total & (1 << to))) {
						if (is_capture == false) {
							counter = 0;

							is_capture = true;
						}
						add_new_capture(
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

			// to do inne dla przeciwnego koloru
			if (is_capture == false) {
				with = NEIGHBOURS[i][0];
				if (with >= 0 && with < 32) {
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
				with = NEIGHBOURS[i][1];
				if (with >= 0 && with < 32) {
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

	uint8_t from, dire;
	unpack_u16_move(move_packed, from, with, to, dire);
	if (is_capture) {
		printf("Capture move packed: %u\n %d %d %d dir %d\n", move_packed, from, with, to, dire);
	}
	else {
		printf("Normal move packed: %u\n %d %d %d\n", move_packed, from, with, to);
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

	Board board = firstRow();
	printBoard(board);
    
	calcAllMovesCuda(board);


    return 0;
}


