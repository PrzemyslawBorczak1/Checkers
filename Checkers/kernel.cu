
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// czarne na gorze
// uklad 
// notacja 
//   a b c d e f g h
// 
//		7	3
//		6	2
//	..	5	1
//	8	4	0
struct Board {
    uint32_t white_pawns;
    uint32_t white_kings;
    uint32_t black_pawns;
    uint32_t black_kings;

	uint32_t occupied_white;
	uint32_t occupied_black;

	uint32_t occupied_total;

	char white_strength;
	char black_strength;

	char is_white_move;
	char is_capture;
};


__host__ __device__ void writeBoardToBuff(char buffer[72], uint32_t board, char c) {
	int x = 0;
	int y = 1;
	int pos = 0;
	for (int i = 0; i < 32; i++) {
		if (board & (1u << (31 - i))) {
			pos = y * 9 + x;
			buffer[pos] = c;
		}
		y += 2;

		if (y >= 8) {
			y = 1 - (y - 8);
			x++;
		}

	}
}

void printBoard(Board board) {

	char buffer[72];
	memset(buffer, ' ', 72);

	buffer[71] = '\0';

	for (int i = 0; i < 7; i++) {
		buffer[i * 9 + 8] = '\n';
	}


	writeBoardToBuff(buffer, board.black_pawns, 'b');
	writeBoardToBuff(buffer, board.white_pawns, 'w');
	writeBoardToBuff(buffer, board.black_kings, 'B');
	writeBoardToBuff(buffer, board.white_kings, 'V');



	printf("\n");
	printf("   a b c d e f g h\n");
	printf("  ----------------\n");

	for (int row = 0; row < 8; row++) {
		printf("%d| ", 8 - row);
		for (int col = 0; col < 8; col++) {
			int pos = row * 9 + col;
			printf("%c ", buffer[pos]);
		}
		printf("\n");
	}

	printf("  ----------------\n");
	printf("   a b c d e f g h\n\n");

}

__host__ __device__ void print_int(uint32_t n) {
	char buffer[72];
	memset(buffer, '_', 72);

	buffer[71] = '\0';

	for (int i = 0; i < 7; i++) {
		buffer[i * 9 + 8] = '\n';
	}


	writeBoardToBuff(buffer, n, '*');
	printf("%s\n", buffer);
}

__host__ __device__ void print_bin(uint32_t n)
{
	for (int i = 31; i >= 0; i--) {
		if (n & (1u << i)) {
			printf("1");
		}
		else {
			printf("0");
		}
		if (i % 4 == 0) {
			printf(" ");
		}
	}
	printf("\n");
}


Board startBoard() {
	Board b;
	b.white_pawns = 0x31313131;
	b.white_kings = 0;
	b.black_pawns = 0x8C8C8C8C;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;
	

	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board captureBoard() {
	
	Board b;
	b.white_pawns = 0x31313131;
	b.white_kings = 0;
	b.black_pawns = 0x8A8A8A8A;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;

}

Board endBoard() {

	Board b;
	b.white_pawns = 0x80808080;
	b.white_kings = 0;
	b.black_pawns = 0x08080808;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;

}

Board backCapture() {
	Board b;
	b.white_pawns = 0x02020202;
	b.white_kings = 0;
	b.black_pawns = 0x20202020;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board firstRow() {
	Board b;
	b.white_pawns = 0x10101010;
	b.white_kings = 0;
	b.black_pawns = 0x08080808;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board lastRow1() {
	Board b;
	b.white_pawns = 0x08080808;
	b.white_kings = 0;
	b.black_pawns = 0x01010101;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board lastRow2() {
	Board b;
	b.white_pawns = 0x08080808;
	b.white_kings = 0;
	b.black_pawns = 0x10101010;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board cycleBoard1() {
	Board b;
	b.white_pawns = 0x00000100;
	b.white_kings = 0;
	b.black_pawns = 0x00C0A060;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}

Board cycleBoard2() {
	Board b;
	b.white_pawns = 0x00000800;
	b.white_kings = 0;
	b.black_pawns = 0x0060A0C0;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;


	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}



__device__ bool checkCaptureForWhite(Board* b, char from, char with, char to, uint32_t occupied_total, uint32_t occupied_black) {
	// chyba bez?
	/*if(with < 0 || with >= 32) {
		return;
	}*/

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


__device__ bool findCapturesForWhite(Board* b, char index, char offset, uint32_t occupied_total, uint32_t occupied_black, char depth) {
	

	// prawa gora
	char with = index - 4 + offset;
	if (checkCaptureForWhite(b, index, with, index - 7, occupied_total, occupied_black)) {
		findCapturesForWhite(b, index - 7, offset, occupied_total, occupied_black ^ (1 << with), depth + 1);

		printf("%d Capture with %d to %d depth: %d\n", index, with, index - 7, depth);
	}

	// lewa gora
	with = index + 4 + offset;
	if (checkCaptureForWhite(b, index, with, index + 9, occupied_total, occupied_black)) {
		findCapturesForWhite(b, index + 9, offset, occupied_total, occupied_black ^ (1 << with), depth + 1);


		printf("%d Capture with %d to %d depth: %d\n", index, with, index + 9, depth);
	}

	// prawy tyl
	with = index - 5 + offset;
	if (checkCaptureForWhite(b, index, with, index - 9, occupied_total, occupied_black)) {
		findCapturesForWhite(b, index - 9, offset, occupied_total, occupied_black ^ (1 << with), depth + 1);


		printf("%d Capture with %d to %d depth: %d\n", index, with, index - 9, depth);
	}

	// lewy tyl
	with = index + 3 + offset;
	if (checkCaptureForWhite(b, index, with, index + 7, occupied_total, occupied_black)) {
		findCapturesForWhite(b, index + 7, offset, occupied_total, occupied_black ^ (1 << with), depth + 1);


		printf("%d Capture with %d to %d depth: %d\n", index, with, index + 7, depth);
	}

}

__global__ void checkersKernel(Board* b,  char* ret)
{

	__shared__ char board[32];

	int index = threadIdx.x;
	board[index] = 0;
	int x = index / 4;
	int y = (index % 4) ;
	int offset = 1 -  x % 2;
	char with;

	while (true)
	{
		switch (b->is_white_move) {
		case true:
			if (!(b->occupied_white & 1 << index)) {
				break;
			}


			with = index - 4 + offset;
			if (((with / 4) % 2) != (index / 4) % 2) {// nie mozna zlaczyc bo on jest zwiazny z drugim warunkiem
				if (with > 0 && with < 32) {
					if (!(b->occupied_total & (1 << with))) {
						board[index]++;
					}
				}
			}

			with = index + 4 + offset;
			if (((with / 4) % 2) != (index / 4) % 2) {
				if (with > 0 && with < 32) {
					if (!(b->occupied_total & (1 << with))) {
						board[index]++;
					}
				}
			}
			
			
			findCapturesForWhite(b, index, offset, b->occupied_total ^ (1 << index), b->occupied_black, 0);
			

			printf("%d : board: %d\n", index, board[index]);

			break;

		case false:

			break;

		}
		return;
	}
}

cudaError_t calcAllMovesCuda(Board board_in) {

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	/*printf("Board: %d", sizeof(Board));
	printf("Inside: %d", 6 * sizeof(uint32_t) + 3 * sizeof(char));
	if (sizeof(Board) != 6 * sizeof(uint32_t) + 3 * sizeof(char)) {
		printf("Bad size of board");
		goto Error;
	}*/


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

	checkersKernel << <1, 32 >> > (d_in, d_ret);

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
	Board board = cycleBoard2();
	printBoard(board);
    
	calcAllMovesCuda(board);


    return 0;
}


