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
constexpr int TT_SIZE = 1 << 26;

// 空格, 小兵, 騎士, 主教, 城堡, 皇后, 國王
// 隨便寫的，給 MVV 湊合用
inline const int MVV_VALUES[7] = {0, 10, 50, 30, 30, 90, 900};

// ==============================================================================
// Machine Learning 來的特徵向量，Sigmoid Function 太神啦啦啦啦
// ==============================================================================

// 這就是 V1
inline const int PIECE_VALUES[7] = {0, 100, 500, 312, 312, 875, 2000000};

inline const int u2[5] = {14, -14, -19, -12, -19}; // 曼哈頓權重

inline const int u3[7][30] = {
    {0}, // 空格
    {   0,    0,    0,    0,    0,    0,    3,   -3,    2,  -11,   35,  -14,   -7,  -10,  -10,    7,  -24,  -28,   31,   30,   21,   -9,   22,    3,  -32,    0,    0,    0,    0,    0},
    {  -2,    0,   -2,  -12,    0,  -14,   -1,   16,   -9,    3,   -8,    1,  -21,    9,   -6,  -28,    2,  -13,    6,    7,  -17,    2,  -18,   -3,  -22,   38,   14,    2,   17,  -33},
    {  -2,   -3,   -3,    0,    3,   -1,   -4,    3,   -7,   -2,   -7,   -2,    3,   -1,   -4,  -23,   10,   -8,    1,    5,    0,   15,    1,   13,   -3,    0,  -28,   -1,  -31,    0},
    {   0,   -1,    0,   -3,    0,    2,    0,    2,    0,   -2,    0,    4,    0,  -19,    0,    6,    0,    1,    0,  -26,    0,    4,    0,    7,    0,    2,    0,  -26,    0,   17},
    {  -1,   -2,   -2,   -4,   -1,   -1,   -5,   -4,  -13,    4,   -7,   -1,   -3,   -1,  -13,    1,   -5,  -10,   -6,   -5,   12,    4,  -32,  -20,   15,   -1,   -9,    7,    4,  -21},
    {  -1,    0,    0,    0,    1,   -1,   -3,    1,    7,    0,    0,   -3,    1,   13,    0,    0,    4,   -1,    7,    1,   -3,   -5,    6,    2,   14,  -20,  -20,    5,   28,   -6}
};

inline const int u4_tactical[6] = {0, 18, 19, 20, 38, 500000}; // 吃子威脅權重
inline const int u5_pawn_stages[4] = {-9, 15, 28, 6}; // 升變獎勵 / 懲罰

// 國王親征權重
inline const int u7_king_step[30] = {-2, 2, 2, 2, 2, -2, 1, 2, 2, 2, 1, 1, 2, 2, 0, 1, 0, 2, 2, 2, 0, 0, 2, 1, 1, -1, 2, 1, 2, 0};
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