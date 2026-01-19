
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


#define MAX_CAPTURE_DEPTH 12

__device__ __constant__ const int8_t WITH_OFFSETS[4] = { -4, 4, -5, 3 };
__device__ __constant__ const int8_t DEST_OFFSETS[4] = { -7, 9, -9, 7 };

struct SearchState {
	char index;
	uint32_t occupied_black;
	uint8_t stage;
};

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



#define HAS_CHILD_BIT 0x80
#define DIR_MASK 0x07
#define LANDING_MASK 0x70 // To track which landing square we are on

//__device__ int getNextSquare(int sq, int dir) {
//	// Standard 32-bit board diagonal sliding
//	// dir: 0=NE(-4/-3), 1=SE(4/5), 2=SW(3/4), 3=NW(-5/-4)
//	int row = sq >> 2;
//	bool isRowEven = (row & 1) == 0;
//
//	int next;
//	if (dir == 0) next = isRowEven ? sq - 4 : sq - 3;
//	else if (dir == 1) next = isRowEven ? sq + 4 : sq + 5;
//	else if (dir == 2) next = isRowEven ? sq + 3 : sq + 4;
//	else next = isRowEven ? sq - 5 : sq - 4;
//
//	// Boundary check: row must change by exactly 1 and be within 0-31
//	int nextRow = next >> 2;
//	if (next < 0 || next >= 32 || abs(nextRow - row) != 1) return -1;
//	return next;
//}


//__device__ int countFlyingKingLeaves(uint32_t startOccTotal, uint32_t startOccBlack, char startIndex) {
//	SearchState stack[MAX_CAPTURE_DEPTH];
//	int sp = 0;
//	int leafCount = 0;
//
//	stack[0].index = startIndex;
//	stack[0].occupied_black = startOccBlack;
//	stack[0].stage = 0; // Bits: [Empty][LandingIdx:3][DirIdx:3] \
//
//	while (sp >= 0) {
//		SearchState* curr = &stack[sp];
//		int dir = curr->stage & DIR_MASK;
//
//		if (dir < 4) {
//			bool foundCaptureInDir = false;
//			int scan = curr->index;
//
//			// 1. Slide to find the first piece in this direction
//			while ((scan = getNextSquare(scan, dir)) != -1) {
//				uint32_t bit = (1 << scan);
//				if (startOccTotal & bit) {
//					// If it's an opponent piece and not already captured in this sequence
//					if (curr->occupied_black & bit) {
//
//						// 2. We found an enemy! Now check landing squares behind it
//						int land = scan;
//						int landingCount = 0;
//						int targetLanding = (curr->stage & LANDING_MASK) >> 4;
//
//						while ((land = getNextSquare(land, dir)) != -1) {
//							if (startOccTotal & (1 << land)) break; // Blocked by another piece
//
//							// If this is the specific landing square we are currently exploring
//							if (landingCount == targetLanding) {
//								curr->stage |= HAS_CHILD_BIT; // Mark parent as branched
//
//								// Push new state
//								sp++;
//								stack[sp].index = land;
//								stack[sp].occupied_black = curr->occupied_black ^ bit;
//								stack[sp].stage = 0;
//
//								// Increment landing index for when we return to the parent
//								curr->stage = (dir) | ((targetLanding + 1) << 4) | (curr->stage & HAS_CHILD_BIT);
//								foundCaptureInDir = true;
//								break;
//							}
//							landingCount++;
//						}
//
//						// If we've exhausted landing squares for this specific enemy
//						if (!foundCaptureInDir) {
//							curr->stage = (dir + 1) | (0 << 4) | (curr->stage & HAS_CHILD_BIT);
//						}
//					}
//					else {
//						// Hit own piece, direction blocked
//						curr->stage = (dir + 1) | (0 << 4) | (curr->stage & HAS_CHILD_BIT);
//					}
//					break;
//				}
//			}
//			// If the diagonal was empty or we reached the edge without hitting anything
//			if (scan == -1) {
//				curr->stage = (dir + 1) | (0 << 4) | (curr->stage & HAS_CHILD_BIT);
//			}
//		}
//		else {
//			// Leaf logic: if this node never successfully pushed a child, it's a leaf
//			if (!(curr->stage & HAS_CHILD_BIT) && sp > 0) {
//				leafCount++;
//			}
//			sp--;
//		}
//	}
//	return leafCount;
//}

#define UP_RIGHT 0
#define UP_LEFT 1
#define DOWN_LEFT 2
#define DOWN_RIGHT 3
#define NONE 4


__device__ __constant__ const int8_t WITH_OFFSETS_CLOCK[4] = { -4, 4, 3, -5 };
__device__ __constant__ const int8_t DEST_OFFSETS_CLOCK[4] = { -7, 9, 7, -9 };

#define NO_SQUARE 67

// faster division
__device__ int getNextSquare(char index, char dir, char offset) {
	char ret = index + WITH_OFFSETS_CLOCK[dir] + offset;
	//printf("Sq: %d\n", ret);
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
	if (curr == NO_SQUARE) {
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
		if (curr == NO_SQUARE) {
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
		printf("%d Check line 1: %d\n",curr, line);
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
		printf("CaptureKing loop end %d %d %d dir %d new offset: %d\n", curr, with, to,dir, work_offset);
		if (curr == NO_SQUARE) {
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
			if (curr == NO_SQUARE) {
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

			if (curr == NO_SQUARE) {
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

			if (curr == NO_SQUARE) {
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

			if (curr == NO_SQUARE) {
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
			
			if (b->white_pawns & 1 << index) {
				char mv = countWhiteCaptureLeaves(b, index, offset, b->occupied_total ^ (1 << index), b->occupied_black);

				printf("index %d mc: %d\n", index, mv);
			}
			printf("%d : board: %d\n", index, board[index]);

			if(b-> white_kings & 1 << index) {
				//printf("Why checking");
				int m = noCaptureKing(b->occupied_total ^ (1 << index), b->occupied_black, index, NONE, offset);

				printf("king %d m: %d\n", index, m);
				//krolowa
			}

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

	// todo delte afte debugging
	size_t cur = 0;
	cudaDeviceGetLimit(&cur, cudaLimitStackSize);
	printf("Current stack: %zu bytes\n", cur);

	// e.g. 64 KB per thread (pick a value and test)
	cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);

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

	Board board = kingCycle();
	printBoard(board);
    
	calcAllMovesCuda(board);


    return 0;
}


