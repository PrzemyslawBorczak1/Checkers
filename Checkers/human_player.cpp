#include "human_player.h"
#include <iostream>


bool HumanPlayer::parseMove(char* move_str, vector<int>& steps, char delim) {
    char* current = move_str;
    steps.clear();

    while (current != nullptr && *current != '\0') {
        char* next_delim = strchr(current, delim);

        char buffer[3] = { 0 };

        size_t len = (next_delim != nullptr) ? (next_delim - current) : strlen(current);

        if (len == 0 || len >= sizeof(buffer)) {
            return false;
        }

        strncpy(buffer, current, len);
        buffer[len] = '\0';

        int idx = 0;
        if (!charToField(buffer, idx)) {
            return false;
        }

        steps.push_back(static_cast<int>(idx));

        if (next_delim != nullptr) {
            current = next_delim + 1;
        }
        else {
            current = nullptr;
        }
    }

    return steps.size() >= 2;
}


vector<int> HumanPlayer::normalizeMove(vector<int>& steps)
{
    const int n = (int)steps.size();
    if (n <= 2) return steps;

    vector<int> out;
    out.reserve(n);
    out.push_back(steps[0]);

    for (int i = 1; i < n - 1; ++i) {
        int d1 = dirFromSteps(steps[i - 1], steps[i]);
        int d2 = dirFromSteps(steps[i], steps[i + 1]);

        if (d1 == d2) continue;

        out.push_back(steps[i]);
    }

    out.push_back(steps[n - 1]);
    return out;
}

bool HumanPlayer::moveEqualsNormalized(vector<int>& a, vector<int>& b)
{
    char buffa[40];
    char buffb[40];
    moveToChar(a, false, buffa);
    moveToChar(b, false, buffb);


    vector<int> na = normalizeMove(a);
    vector<int> nb = normalizeMove(b);
    return na == nb;
}

PossibleMove* HumanPlayer::findMoveNormalized(vector<PossibleMove>& pm, vector<int>& m)
{
    vector<int> nm = normalizeMove(m);

    for (auto& p : pm) {
        if (normalizeMove(p.move) == nm) {
            return &p;
        }
    }
    return nullptr;
}

void HumanPlayer::printMoveVector(vector<PossibleMove> pm) {
    for (auto p : pm) {

        char buff[40];
        moveToChar(p.move, p.is_capture, buff);
        printf("%s\n", buff);
    }
    printf("\n");
}




void HumanPlayer::MakeMove(Board& board, char* ret, int moves_without_progress) {
    ret[0] = '\0';
    char buf[100];
    buf[0] = '\0';

    vector<PossibleMove> possible_moves = getAllMoves(board, player_color);

    if (possible_moves.empty()) {
        return;
    }

    bool is_capture = false;
    if (possible_moves[0].is_capture) {
        is_capture = true;
        printf("Masz bicie!\n");
    }

    char delim = is_capture ? ':' : '-';

    vector<int> steps;



    while (true)
    {
        printf("Mozliwe ruchy:\n");
        printMoveVector(possible_moves);
        printf("Podaj ruch: ");
        fflush(stdout);

        if (!fgets(buf, (int)sizeof(buf), stdin)) {
            ret[0] = '\0';
            return;
        }
        size_t n = strlen(buf);
        while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
            buf[--n] = '\0';
        }

        if (!parseMove(buf, steps, delim)) {
            printf("Nie odpowiedni format ruchu\n");
            continue;
        }


        auto fn = findMoveNormalized(possible_moves, steps);
        if (fn == nullptr) {
            printf("Ruch nie jest dozwolony\n");
            continue;
        }

        board = fn->resulting_board;
        strcat(ret, buf);
        return;

    }
}


