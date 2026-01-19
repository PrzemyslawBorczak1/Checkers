#pragma once
#include <cstdint>

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