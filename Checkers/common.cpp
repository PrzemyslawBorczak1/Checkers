#pragma once
#include "common.h"
#include <stdio.h>


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


Board startBoard() {
    Board b;
    b.white_pawns = 0x31313131;
    b.white_kings = 0;
    b.black_pawns = 0x8C8C8C8C;
    b.black_kings = 0;
    return b;
}

