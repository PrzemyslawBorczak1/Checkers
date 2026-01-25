#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mcts_player.h"
#include "human_player.h"
#include "common.h"


#define GT_HH 0
#define GT_HC 1
#define GT_CC 2

typedef struct {
    const char* file;   // argv[1]
    int time_sec;       // argv[2]
    int game_type;      // argv[3] 0/1/2
    Color color;        // argv[4] 0/1
} Options;

void usage(void) {
    printf(
        "checkers file time game_type color\n"
        "-file       plik zapisu partii (np. game.txt)\n"
        "-time       limit czasu na ruch w sekundach (>0)\n"
        "-game_type  0 = czlowiek-czlowiek, 1 = czlowiek-komputer, 2 = komputer-komputer\n"
        "-color      0 = biale, 1 = czarne  (w trybie 1 to kolor czlowieka)\n"
        "\n"
    );
}

static int parse_int_strict(const char* s, int* out) {
    char* end = NULL;
    long v;

    if (s == NULL || s[0] == '\0') return 0;

    v = strtol(s, &end, 10);
    if (end == NULL || *end != '\0') return 0;

    *out = (int)v;
    return 1;
}

int parse_args(int argc, char** argv, Options* opt) {
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    if (argc != 5) {
        printf("Blad: zla liczba argumentow.\n\n");
        usage();
        return 0;
    }

    opt->file = argv[1];
    if (opt->file == NULL || opt->file[0] == '\0') {
        printf("Blad: file nie moze byc pusty.\n\n");
        usage();
        return 0;
    }

    if (!parse_int_strict(argv[2], &opt->time_sec) || opt->time_sec <= 0) {
        printf("Blad: time musi byc liczba calkowita > 0.\n\n");
        usage();
        return 0;
    }

    if (!parse_int_strict(argv[3], &opt->game_type) ||
        (opt->game_type != 0 && opt->game_type != 1 && opt->game_type != 2)) {
        printf("Blad: game_type musi byc 0, 1 albo 2.\n\n");
        usage();
        return 0;
    }

    int color_int;
    if (!parse_int_strict(argv[4], &color_int) || (color_int != 0 && color_int != 1)) {
        printf("Blad: color musi byc 0 (biale) albo 1 (czarne).\n\n");
        usage();
        return 0;
    }

    opt->color = (color_int == 0) ? Color::WHITE : Color::BLACK;
    return 1; // OK
}

void decide_players(const Options* opt, int* white_is_human, int* black_is_human) {
    if (opt->game_type == GT_HH) {
        *white_is_human = 1;
        *black_is_human = 1;
        return;
    }

    if (opt->game_type == GT_CC) {
        *white_is_human = 0;
        *black_is_human = 0;
        return;
    }

    // GT_HC: kolor w opt->color to kolor czlowieka
    if (opt->color == Color::WHITE) {
        *white_is_human = 1;
        *black_is_human = 0;
    }
    else {
        *white_is_human = 0;
        *black_is_human = 1;
    }
}


#define NO_PROGRESS_LIMIT 30

int main(int argc, char** argv) {
  
    argc = 5;
    const char* argv_const[] = { "checkers", "game.txt", "1", "2", "0" };
    argv = (char**)argv_const;

    Options opt;
    int white_is_human = 0, black_is_human = 0;

    if (!parse_args(argc, argv, &opt)) {
        return 1;
    }

    decide_players(&opt, &white_is_human, &black_is_human);

    printf("Game starts:\n");
    printf(" file=%s\n", opt.file);
    printf(" time=%d\n", opt.time_sec);
    printf(" game_type=%d\n", opt.game_type);
    printf(" color=%d\n", (int)opt.color);
    printf(" white_is_human=%d black_is_human=%d\n", white_is_human, black_is_human);

    Board board = kingLine4();

    Player* white_player;
    if(white_is_human)
        white_player = new HumanPlayer(Color::WHITE);
	else
		white_player = new MCTSPlayer(Color::WHITE, opt.time_sec);

    Player* black_player;
    if(black_is_human)
        black_player = new HumanPlayer(Color::BLACK);
	else
		black_player = new MCTSPlayer(Color::BLACK, opt.time_sec);

    Color side_to_move = Color::WHITE;
    char* move_str;
    Color loser;
	int no_progress_count = 0;
    while (1) {

		printf("\n\n=====================================\n");
		printf("No progress count: %d\n", no_progress_count);
        if (side_to_move == Color::WHITE)
            printf("White to move:\n");
        else
            printf("Black to move:\n");
		printBoard(board);
		Board old_board = board;
		int old_white_pawn_count = __popcnt(board.white_pawns);
		int old_black_pawn_count = __popcnt(board.black_pawns);

        if (side_to_move == Color::WHITE)
            move_str = white_player->MakeMove(board);
        else
            move_str = black_player->MakeMove(board);


        // no move
        if(move_str[0] == '\0') {
			loser = side_to_move; 
            printf("No moves: %s\n", move_str);
            break;
		}


		// no progress 
        int white_pawn_count = __popcnt(board.white_pawns);
        int black_pawn_count = __popcnt(board.black_pawns);

        if (old_board.white_pawns == board.white_pawns &&
            old_board.black_pawns == board.black_pawns &&
            old_white_pawn_count == white_pawn_count &&
            old_black_pawn_count == black_pawn_count
            ) {
            no_progress_count++;
            if (no_progress_count > NO_PROGRESS_LIMIT) {
                loser = side_to_move;
                printf("%d Move without progress: %s\n", no_progress_count, move_str);
                break;
            }
        }
        else {
            no_progress_count = 0;
        }


        printf("Move: %s\n", move_str);

        side_to_move = (side_to_move == Color::WHITE) ? Color::BLACK : Color::WHITE;
    }

    if (loser == Color::WHITE)
		printf("Black wins!\n");
    else
        printf("White wins!\n");

    return 0;
}
