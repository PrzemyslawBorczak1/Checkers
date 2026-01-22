#pragma once
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tests.h"

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


__host__ __device__ void printBoard(uint32_t black_pawns, uint32_t white_pawns, uint32_t black_kings, uint32_t white_kings) {

	char buffer[72];
	memset(buffer, ' ', 72);

	buffer[71] = '\0';

	for (int i = 0; i < 7; i++) {
		buffer[i * 9 + 8] = '\n';
	}


	writeBoardToBuff(buffer, black_pawns, 'b');
	writeBoardToBuff(buffer, white_pawns, 'w');
	writeBoardToBuff(buffer, black_kings, 'B');
	writeBoardToBuff(buffer, white_kings, 'V');



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


__host__ __device__ void printBoard(Board board) {

	printBoard(board.black_pawns, board.white_pawns, board.black_kings, board.white_kings);

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

Board cycleBoard3() {
	Board b;
	b.white_pawns = 0x00008000;
	b.white_kings = 0;
	b.black_pawns = 0x00060600;
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

Board kingMainLine1() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00000008;
	b.black_pawns = 0x00000080;
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

Board kingMainLine2() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x10000000;
	b.black_pawns = 0x00000000;
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

Board kingLine1() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00010000;
	b.black_pawns = 0x00000000;
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

Board kingLine2() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00010000;
	b.black_pawns = 0x00002000;
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

Board kingLine3() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00000080;
	b.black_pawns = 0x01402400;
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

Board kingAllDir() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00004000;
	b.black_pawns = 0x00060600;
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

Board dr() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x80000000;
	b.black_pawns = 0x04000000;
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
Board dl() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00000008;
	b.black_pawns = 0x00000080;
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
Board ur() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x10000000;
	b.black_pawns = 0x01000000;
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
Board ul() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00000001;
	b.black_pawns = 0x00000020;
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
Board kingCycle() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x00000800;
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


Board kingWierd() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x01000000;
	b.black_pawns = 0x02246420;
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

Board kingBlocking1() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x10000000;
	b.black_pawns = 0x04646700;
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

Board kingBlocking2() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x10000000;
	b.black_pawns = 0x04446700;
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

Board kingMultiple() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x10001000;
	b.black_pawns = 0x00068240;
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


Board kingPawnsCapture() {
	Board b;
	b.white_pawns = 0x22301000;
	b.white_kings = 0x10000001;
	b.black_pawns = 0x00068240;
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

Board kingPawnsNoCapture() {
	Board b;
	b.white_pawns = 0x22301000;
	b.white_kings = 0x10000001;
	b.black_pawns = 0x0004C240;
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


Board kingPawnsBlack() {
	Board b;
	b.white_pawns = 0x00068240;
	b.white_kings = 0;
	b.black_pawns = 0x22301000;
	b.black_kings = 0x10000001;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;



	b.white_strength = 12;
	b.black_strength = 12;
	b.is_white_move = true;
	b.is_capture = false;

	return b;
}


Board promotion() {
	Board b;
	b.white_pawns = 0x04040404;
	b.white_kings = 0;
	b.black_pawns = 0x22222222;
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


Board kingLine4() {
	Board b;
	b.white_pawns = 0;
	b.white_kings = 0x80000000;
	b.black_pawns = 0x04021013;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;

	return b;
}

Board captureNoPromotion() {
		Board b;
	b.white_pawns = 0x00000404;
	b.white_kings = 0;
	b.black_pawns = 0x00008080;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;
	return b;
}

Board capturePromotion() {
	Board b;
	b.white_pawns = 0x01826400;
	b.white_kings = 0;
	b.black_pawns = 0x44200180;
	b.black_kings = 0;
	b.occupied_white = b.white_pawns | b.white_kings;
	b.occupied_black = b.black_pawns | b.black_kings;
	b.occupied_total = b.occupied_white | b.occupied_black;
	return b;
}