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
        " file       plik zapisu partii (np. game.txt)\n"
        " time       limit czasu na ruch w sekundach (>0)\n"
        " game_type  0 = czlowiek-czlowiek, 1 = czlowiek-komputer, 2 = komputer-komputer\n"
        " color      0 = biale, 1 = czarne  (w trybie 1 to kolor czlowieka)\n"
        "\n"
    );
}

// funkcja pomocnicza do parsowania argumentow komendy 
int parseInt(const char* s, int* out) {
    char* end = NULL;
    long v;

    if (s == NULL || s[0] == '\0')
        return 0;

    v = strtol(s, &end, 10);
    if (end == NULL || *end != '\0') 
        return 0;

    *out = (int)v;
    return 1;
}

int parseArgs(int argc, char** argv, Options* opt) {

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

    if (!parseInt(argv[2], &opt->time_sec) || opt->time_sec <= 0) {
        printf("Blad: time musi byc liczba calkowita > 0.\n\n");
        usage();
        return 0;
    }

    if (!parseInt(argv[3], &opt->game_type) ||
        (opt->game_type < 0 || opt->game_type > 2)) {
        printf("Blad: game_type musi byc 0, 1 albo 2.\n\n");
        usage();
        return 0;
    }

    int color_int;
    if (!parseInt(argv[4], &color_int) || (color_int != 0 && color_int != 1)) {
        printf("Blad: color musi byc 0 (biale) albo 1 (czarne).\n\n");
        usage();
        return 0;
    }

    opt->color = (color_int == 0) ? Color::WHITE : Color::BLACK;
    return 1;
}

// zapis ruchow do pliku podanego w wywolaniu
bool saveMovesToFile(const char* filename, const char* all_moves_str, int len)
{
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    bool ok = (fwrite(all_moves_str, 1, (size_t)len, f) == (size_t)len);
    if (!(fputc('\n', f) != EOF)) {
        ok = false;
    }

    fclose(f);
    return ok;
}

// wybor ktory gracz jest czlowiekiem a ktory komputerem (w przypadku gry czlowiek vs komputer color decyduje ktora strona bedzie gracz)
void decidePlayer(const Options* opt, bool* white_is_human, bool* black_is_human) {
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


    if (opt->color == Color::WHITE) {
        *white_is_human = 1;
        *black_is_human = 0;
    }
    else {
        *white_is_human = 0;
        *black_is_human = 1;
    }
}

// dopisanie ruchu do lancucha wszystkich ruchow
void appendMove(char* all_moves_str, int* len, char* move_str)
{
    int m = (int)strlen(move_str);

    if (*len > 0) {
        all_moves_str[(*len)++] = ' ';
    }

    memcpy(all_moves_str + *len, move_str, (size_t)m);
    *len += m;

    all_moves_str[*len] = '\0';
}

// wydruk ustawien gry
void printGameSettings(Options opt, bool white_is_human, bool black_is_human) {
    printf("\n=== Checkers: start gry ===\n");
    printf("  plik zapisu : %s\n", opt.file);
    printf("  czas/ruch   : %d s\n", opt.time_sec);
    printf("  tryb gry    : %d (%s)\n",
        opt.game_type,
        (opt.game_type == 0) ? "czlowiek-czlowiek" :
        (opt.game_type == 1) ? "czlowiek-komputer" :
        "komputer-komputer");
    printf("  kolor gracza: %d (%s)\n",
        (int)opt.color,
        ((int)opt.color == 0) ? "biale" : "czarne");
    printf("  sterowanie  : biale=%s, czarne=%s\n",
        white_is_human ? "czlowiek" : "komputer",
        black_is_human ? "czlowiek" : "komputer");
    printf("===========================\n\n");
}


int main(int argc, char** argv) {
    
    // todo usunac
    argc = 5;
    const char* argv_const[] = { "checkers", "game.txt", "1", "2", "0" };
    argv = (char**)argv_const;

    Options opt;
    bool white_is_human = false, black_is_human = false;

    if (!parseArgs(argc, argv, &opt)) {
        return 1;
    }

    decidePlayer(&opt, &white_is_human, &black_is_human);
	printGameSettings(opt, white_is_human, black_is_human);

    Board board = startBoard();

    Player* white_player = white_is_human ?
        (Player*)new HumanPlayer(Color::WHITE) :
		(Player*)new MCTSPlayer(Color::WHITE, opt.time_sec);


    Player* black_player = black_is_human ?
		(Player*)new HumanPlayer(Color::BLACK) :
		(Player*)new MCTSPlayer(Color::BLACK, opt.time_sec);


    Color side_to_move = Color::WHITE;

    char move_str[40];
	char all_moves_str[2048];
	all_moves_str[0] = '\0';
	int moves_len = 0;

    Color loser;
	int no_progress_count = 0;
    while (1) {

        printf("\n\n=====================================\n");
        printf("Ruchy bez postepu: %d\n", no_progress_count);

        printf("Partia: %s\n", all_moves_str);

        side_to_move == Color::WHITE ? printf("Tura bialego:\n") : printf("Tura czarnego:\n");

        printBoard(board);

        Board old_board = board;
        int old_white_pawn_count = __popcnt(board.white_pawns);
        int old_black_pawn_count = __popcnt(board.black_pawns);

        side_to_move == Color::WHITE ?
            white_player->MakeMove(board, move_str, no_progress_count) :
            black_player->MakeMove(board, move_str, no_progress_count);


		// brak ruchu ktorejs ze stron - koniec gry
        if (move_str[0] == '\0') {
            loser = side_to_move;
            break;
        }


		// sprawdzenie braku postepu (brak bicia i ruchow pionkow)
        int white_pawn_count = __popcnt(board.white_pawns);
        int black_pawn_count = __popcnt(board.black_pawns);

        if (old_board.white_pawns == board.white_pawns &&
            old_board.black_pawns == board.black_pawns &&
            old_white_pawn_count == white_pawn_count &&
            old_black_pawn_count == black_pawn_count
            ) {
            no_progress_count++;
            if (no_progress_count > NO_PROGRESS_LIMIT) {
                loser = Color::UNDEFINED;
                break;
            }
        }
        else {
            no_progress_count = 0;
        }

        appendMove(all_moves_str, &moves_len, move_str);
        side_to_move = (side_to_move == Color::WHITE) ? Color::BLACK : Color::WHITE;

    }

    if (loser == Color::WHITE)
		printf("Wygrana czarnego!\n");
	else if (loser == Color::BLACK)
        printf("Wygrana bialego!\n");
    else
        printf("Remis!\n");

	if(!saveMovesToFile(opt.file, all_moves_str, moves_len))
		printf("Blad zapisu parti do pliku: %s\n", opt.file);

    return 0;
}
