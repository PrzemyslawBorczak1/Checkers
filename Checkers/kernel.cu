
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tests.h"


#define MAX_CAPTURE_DEPTH 12

__device__ __constant__ const int8_t WITH_OFFSETS[4] = { -4, 4, -5, 3 };
__device__ __constant__ const int8_t DEST_OFFSETS[4] = { -7, 9, -9, 7 };

struct SearchState {
	char index;
	uint32_t occupied_black;
	uint8_t stage;
};

__device__ bool checkCaptureForWhite(Board* b, char from, char with, char to, uint32_t occupied_total, uint32_t occupied_black) {
	

	if (to < 0 || to >= 32) {
		return false;
	}

	if (from / 4 % 2 != to / 4 % 2) {
		return false;
	}

	if ((occupied_black & (1 << with))) {
		if (!(occupied_total & (1 << to))) {
			return true;
		}
	}
	return false;
}


#define HAS_CHILD_BIT 0x80
#define DIRECTION_MASK 0x0F

__device__ int countWhiteCaptureLeaves(Board* b, char startIndex, char offset, uint32_t startOccupiedTotal, uint32_t startOccupiedBlack) {
	SearchState stack[MAX_CAPTURE_DEPTH];
	int sp = 0;
	int leafCount = 0;

	stack[0].index = startIndex;
	stack[0].occupied_black = startOccupiedBlack;
	stack[0].stage = 0; 

	while (sp >= 0) {
		SearchState* current = &stack[sp];
		int dir = (current->stage & DIRECTION_MASK);

		if (dir < 4) {
			current->stage++;

			char idx = current->index;
			char with = idx + WITH_OFFSETS[dir] + offset;
			char dest = idx + DEST_OFFSETS[dir];

			if (checkCaptureForWhite(b, idx, with, dest, startOccupiedTotal, current->occupied_black)) {
				current->stage |= HAS_CHILD_BIT;

				sp++;
				if (sp < MAX_CAPTURE_DEPTH) {
					stack[sp].index = dest;
					stack[sp].occupied_black = current->occupied_black ^ (1 << with);
					stack[sp].stage = 0; 
				}
			}
		}
		else {
			if (!(current->stage & HAS_CHILD_BIT) && sp > 0) {
				leafCount++;
			}
			sp--;
		}
	}

	return leafCount;
}






#define UP_RIGHT 0
#define UP_LEFT 1
#define DOWN_LEFT 2
#define DOWN_RIGHT 3
#define NONE 4


__device__ __constant__ const int8_t WITH_OFFSETS_CLOCK[4] = { -4, 4, 3, -5 };

#define NO_SQUARE 67

// faster division
__device__ int getNextSquare(char index, char dir, char offset) {
	char ret = index + WITH_OFFSETS_CLOCK[dir] + offset;
	if (ret < 0 || ret > 31)
		return NO_SQUARE;

	if (ret / 4 % 2 == index / 4 % 2)
		return NO_SQUARE;

	return ret;
}

__device__ int captureKing(uint32_t occ_total, uint32_t black, char index, char dir, char offset);

__device__ int checkLine(uint32_t occ_total, uint32_t black, char index, char dir, char offset) {
	char work_offset = offset;
	char curr = index;
	char with = getNextSquare(curr, dir, offset);
	char to = getNextSquare(with, dir, 1 - offset);
	if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
		return 0;
	}
	while (1) {
		if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {
			printf("Line found capture %d %d %d\n", curr, with, to);
			return captureKing(occ_total ^ (1 << with), black ^ (1 << with), to, dir, work_offset);
		}

		curr = with;
		with = to;
		to = getNextSquare(with, dir, work_offset);
		work_offset = 1 - work_offset;
		if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
			break;
		}
	}
	return 0;
}


__device__ int captureKing(uint32_t occ_total, uint32_t black, char index, char dir, char offset) {
	print_int(occ_total);
	printf("\n");
	print_int(black);

	printf("\n\nCaptureKing call with dir %d\n", dir);
	char work_offset = offset;
	char curr = index;
	char with = getNextSquare(curr, dir, offset);
	char to = getNextSquare(with, dir, 1 - offset);
	bool add_norm = true;

	printf("CaptureKing initial %d %d %d new offset: %d\n", curr, with, to, work_offset);
	int normal = 0;
	int capture = 0;
	while (1) {
		int line = checkLine(occ_total, black, curr, (dir + 1) % 4, work_offset);
		printf("%d Check line 1: %d\n", curr, line);
		if (line != 0) {
			capture += line;
			normal = 0;
			add_norm = false;
		}
		line = checkLine(occ_total, black, curr, (dir + 3) % 4, work_offset);
		printf("%d Check line 2: %d\n", curr, line);
		if (line != 0) {
			capture += line;
			add_norm = false;
			normal = 0;

		}
		printf("%d Normal increment\n", curr);
		normal++;

		printf("CaptureKing check %d %d %d\n", curr, with, to);
		/*	print_int(occ_total);
			printf("\n");
			print_int(black);*/
		if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {


			printf("CaptureKing found capture %d %d %d\n", curr, with, to);


			occ_total &= ~(1 << with);
			black &= ~(1 << with);

			normal = 0;
			curr = with;
			with = to;
			to = getNextSquare(with, dir, work_offset);
			add_norm = true;

			work_offset = 1 - work_offset;


			print_int(occ_total);
			printf("\n");
			print_int(black);

			printf("CaptureKing capture skip to %d %d %d\n", curr, with, to);
		}

		curr = with;
		with = to;
		to = getNextSquare(with, dir, work_offset);
		work_offset = 1 - work_offset;
		printf("CaptureKing loop end %d %d %d dir %d new offset: %d\n", curr, with, to, dir, work_offset);
		if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
			break;
		}
	}
	printf("CaptureKing Capture: %d  Normal: %d\n\n", capture, normal);
	if (add_norm)
	{
		return capture + normal;
	}

	if (capture == 0)
		return normal;
	return capture;


	printf("Capture king %d\n", index);
}


__device__ int noCaptureKing(uint32_t occ_total, uint32_t black, char index, char dir, char offset) {
	int capture = 0;
	int normal = 0;


	if (dir != DOWN_LEFT) {
		char curr = index;
		char with = getNextSquare(curr, UP_RIGHT, offset);
		char to = getNextSquare(with, UP_RIGHT, 1 - offset);
		char work_offset = offset;
		while (1) {


			if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {

				printf("Found right up %d %d %d\n", curr, with, to);
				capture += captureKing(occ_total ^ (1 << with), black ^ (1 << with), to, UP_RIGHT, work_offset);

				break;
			}

			curr = with;
			with = to;
			to = getNextSquare(with, UP_RIGHT, work_offset);
			work_offset = 1 - work_offset;
			if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
				break;
			}
			normal++;
			printf("Right up (first acutall field): %d  %d  %d\n", curr, with, to);

		}
	}

	if (dir != UP_RIGHT) {
		char curr = index;
		char with = getNextSquare(curr, DOWN_LEFT, offset);
		char to = getNextSquare(with, DOWN_LEFT, 1 - offset);
		char work_offset = offset;
		while (1) {

			if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {

				printf("Found left down capture %d %d %d\n", curr, with, to);
				capture += captureKing(occ_total ^ (1 << with), black ^ (1 << with), to, DOWN_LEFT, work_offset);

				break;
			}

			curr = with;
			with = to;
			to = getNextSquare(with, DOWN_LEFT, work_offset);
			work_offset = 1 - work_offset;

			if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
				break;
			}

			normal++;
			printf("Down left (first acutall field): %d  %d  %d\n", curr, with, to);

		}
	}

	if (dir != UP_LEFT) {
		char curr = index;
		char with = getNextSquare(curr, DOWN_RIGHT, offset);
		char to = getNextSquare(with, DOWN_RIGHT, 1 - offset);
		char work_offset = offset;
		while (1) {

			if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {

				printf("Found right down capture %d %d %d\n", curr, with, to);
				capture += captureKing(occ_total ^ (1 << with), black ^ (1 << with), to, DOWN_RIGHT, work_offset);

				break;
			}

			curr = with;
			with = to;
			to = getNextSquare(with, DOWN_RIGHT, work_offset);
			work_offset = 1 - work_offset;

			if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
				break;
			}
			normal++;
			printf("Down right (first acutall field): %d  %d  %d\n", curr, with, to);

		}
	}

	if (dir != DOWN_RIGHT) {
		char curr = index;
		char with = getNextSquare(curr, UP_LEFT, offset);
		char to = getNextSquare(with, UP_LEFT, 1 - offset);
		char work_offset = offset;
		while (1) {

			if (checkCaptureForWhite(NULL, curr, with, to, occ_total, black)) {

				printf("Found left up capture %d %d %d\n", curr, with, to);
				capture += captureKing(occ_total ^ (1 << with), black ^ (1 << with), to, UP_LEFT, work_offset);

				break;
			}

			curr = with;
			with = to;
			to = getNextSquare(with, UP_LEFT, work_offset);
			work_offset = 1 - work_offset;

			if (curr == NO_SQUARE || (occ_total & (1 << curr))) {
				break;
			}

			normal++;
			printf("up left (first acutall field): %d  %d  %d\n", curr, with, to);

		}
	}



	if (capture == 0)
		return normal;
	return capture;
}














__device__ __forceinline__ void unpack_u16_move(
	uint16_t packed,
	uint8_t& from,
	uint8_t& with,
	uint8_t& to
) {
	from = (uint8_t)(packed & 31u);
	with = (uint8_t)((packed >> 5) & 31u);
	to = (uint8_t)((packed >> 10) & 31u);
}


__device__ __forceinline__ void add_new_capture(
	uint16_t& capture,
	uint32_t& rng_reg,
	uint8_t& count,
	uint8_t from,
	uint8_t with,
	uint8_t to
) {
	count++;
	uint32_t x = rng_reg;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_reg = x;
	if ((x % (uint32_t)count) == 0u) {
		capture = (uint16_t)0 | (((to & 31u) << 10) |
			((with & 31u) << 5) |
			((from & 31u)));
	}
}

__device__ __forceinline__ void add_new_normal(
	uint16_t& capture,
	uint32_t& rng_reg,
	uint8_t& count,
	uint8_t from,
	uint8_t to
) {
	count++;
	uint32_t x = rng_reg;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_reg = x;
	if ((x % (uint32_t)count) == 0u) {
		capture = (uint16_t)0 | (((to & 31u) << 10) |
			((from & 31u)));
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
	uint16_t move_packed;

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

	for (int i = 0; i < 32; i++) {
		if (i % 4 == 0) {
			offset = 1 - offset;
		}

		if (!((player_pawns | player_kings) & (1 << i))) {
			continue;
		}

		if (player_pawns & 1 << i) {
			for (char j = 0; j < 4; j++) {
				with = i + WITH_OFFSETS[j] + offset;
				to = i + DEST_OFFSETS[j];

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
						counter++;
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
			if (is_capture == false) {
				with = i - 4 + offset;
				if (with >= 0 && with < 32) {
					if (!(occupied_total & (1 << with))) {
						counter++;
						add_new_normal(
							move_packed,
							seed,
							counter,
							(uint8_t)i,
							(uint8_t)with
						);
					}
				}
				with = i + 4 + offset;
				if (with >= 0 && with < 32) {
					if (!(occupied_total & (1 << with))) {
						counter++;
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
			printf("King");
		}

		board_index++;
	}

	printf("Board counter: %d\n", counter);

	uint8_t from;
	unpack_u16_move(move_packed, from, with, to);
	if (is_capture) {
		printf("Capture move packed: %u\n %d %d %d\n", move_packed, from, with, to);
	}
	else {
		printf("Normal move packed: %u\n %d %d %d\n", move_packed, from, with, to);




		//with = index - 4 + offset;
		//if (((with / 4) % 2) != (index / 4) % 2) {// nie mozna zlaczyc bo on jest zwiazny z drugim warunkiem
		//	if (with > 0 && with < 32) {
		//		if (!(b->occupied_total & (1 << with))) {
		//			board[index]++;
		//		}
		//	}
		//}

		//with = index + 4 + offset;
		//if (((with / 4) % 2) != (index / 4) % 2) {
		//	if (with > 0 && with < 32) {
		//		if (!(b->occupied_total & (1 << with))) {
		//			board[index]++;
		//		}
		//	}
		//}

		//if (b->white_pawns & 1 << index) {
		//	char mv = countWhiteCaptureLeaves(b, index, offset, b->occupied_total ^ (1 << index), b->occupied_black);

		//	printf("index %d mc: %d\n", index, mv);
		//}
		//printf("%d : board: %d\n", index, board[index]);

		//if (b->white_kings & 1 << index) {
		//	printf("White king checking");
		//	int m = noCaptureKing(b->occupied_total ^ (1 << index), b->occupied_black, index, NONE, offset);

		//	printf("king %d m: %d\n", index, m);
		//	//krolowa
		//}

	}
}

cudaError_t calcAllMovesCuda(Board board_in) {

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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

	Board board = backCapture();
	printBoard(board);
    
	calcAllMovesCuda(board);


    return 0;
}


