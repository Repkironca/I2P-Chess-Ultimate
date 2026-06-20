#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include <chrono>

namespace Logistic {

constexpr int INF_MAX = 1000000000; // Alpha / Beta 邊界，1e9
constexpr int TT_MOVE_SCORE  = 800000000; // TT 命中，8e8
constexpr int CAPTURE_BASE   = 100000000; // 吃子步基礎分，1e8
constexpr int PROMOTE_BASE   = 50000000;  // 升變基礎分，5e7
constexpr int KILLER_BASE    = 20000000;  // 殺手步基礎分，2e7
constexpr int KILLER_PENALTY = 5000000;   // 舊的殺手步扣減分，5e6
constexpr int NORMAL_MOVE    = 0;        // 安靜步

// 系統運算參數
constexpr int MAX_PLY = 128;
constexpr int NUM_KILLERS = 2;

// 本機測試時開 22 就好，實際上可以開到 23，甚至能上 24
constexpr int TT_SIZE = 1 << 22;

// 空格, 小兵, 騎士, 主教, 城堡, 皇后, 國王
// 隨便寫的，給 MVV 湊合用
inline const int MVV_VALUES[7] = {0, 10, 50, 30, 30, 90, 900};

// ==============================================================================
// Machine Learning 來的特徵向量，Sigmoid Function 太神啦啦啦啦
// ==============================================================================

// 這就是 V1
inline const int PIECE_VALUES[7] = {0, 100, 500, 312, 312, 875, 2000000};

inline const int u2[5] = {14, -14, -19, -15, -19}; // 曼哈頓權重

inline const int u3[7][30] = {
    {0}, // 空格
    {   0,    0,    0,    0,    0,   -1,    5,   -3,    3,  -13,   38,   -9,  -15,  -10,  -11,    5,  -25,  -28,   30,   35,   30,   -7,   10,   17,  -36,    0,    0,    0,    0,    0},
    {  -4,    0,   -3,  -14,    1,  -15,   -1,   17,  -11,    2,  -10,    1,  -22,    8,   -7,  -31,    1,  -15,    6,    8,  -18,    2,  -20,   -3,  -21,   38,   15,   -2,   20,  -38},
    {  -2,   -3,   -3,    0,    4,   -1,   -4,    4,   -7,   -2,   -7,   -1,    4,    0,   -5,  -22,   11,   -3,   -1,    7,    0,   20,    2,   16,   -3,    0,  -30,   -1,  -35,    0},
    {   0,   -2,    0,   -4,    0,    5,    0,    2,    0,   -3,    0,    3,    0,  -20,    0,    6,    0,    3,    0,  -29,    0,    1,    0,    4,    0,    2,    0,  -20,    0,   19},
    {  -1,   -2,   -2,   -3,   -1,   -2,   -6,   -5,  -15,    6,   -4,   -2,   -4,   -1,  -15,    1,   -5,  -10,   -6,   -3,    9,    2,  -33,  -16,   17,   -2,   -6,    9,    7,  -23},
    {  -1,    0,    1,    0,    2,   -1,   -3,    1,    8,    0,    0,   -3,    0,   15,    1,    0,    5,   -1,    7,    2,   -3,   -6,    6,    3,   19,  -28,  -24,    6,   33,   -5}
};

inline const int u4_tactical[6] = {0, 20, 10, 21, 38, 500000}; // 吃子威脅權重
inline const int u5_pawn_stages[4] = {-10, 12, 26, 16}; // 升變獎勵 / 懲罰

// 國王親征權重
inline const int u7_king_step[30] = {-2, 2, 2, 2, 2, -1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 0, 2, 2, 2, 0, 1, 2, 1, 1, -1, 2, 1, 2, 0};
constexpr int u_intercept = 0;

// ==============================================================================

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        return p;
    }
};

class Policy {
public:
    static int eval_ctx(State *state, int depth, GameHistory& history, int ply, SearchContext& ctx, const MMParams& p, int alpha, int beta);
    static int q_search(State *state, GameHistory& history, int ply, SearchContext& ctx, const MMParams& p, int alpha, int beta);
    static SearchResult search(State *state, int depth, GameHistory& history, SearchContext& ctx);

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};

} // namespace DuckyQuack