#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include <chrono>

namespace DuckyQuack {

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
inline const int MVV_VALUES[7] = {0, 20, 70, 80, 60, 200, 900};

// ==============================================================================
// Machine Learning 來的特徵向量
// ==============================================================================

// OUTPUT_SCALE = 109.75
// Max Theoretical Eval = 15,000,000
inline const int V1_OFFICIAL[7] = {0, 2, 7, 8, 6, 20, 0};
inline const int V1[7] = {0, 75, 223, 138, 195, 207, 23};

// V2: PST 陣型評估
inline const int V2[7][30] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {     0,      0,      0,      0,      0,     77,     58,     32,     21,    -26,     27,      0,     13,     -4,    -16,      5,      8,    -21,      2,    -39,      4,      0,    -29,    -34,    -32,      0,      0,      0,      0,      0},
    {    26,     50,     44,    -31,    128,     60,     48,     63,    -22,     47,     -7,     82,      8,      0,    -26,     -9,     33,     -8,      3,      3,      3,    -37,      2,    -43,     32,     14,     -9,    -11,     37,    -31},
    {     0,     45,      0,     30,      0,    -13,      0,     64,      0,    -20,      0,     98,      0,     52,      0,     22,      0,     29,      0,     12,      0,    -10,      0,    -28,      0,    -57,      0,    -48,      0,    -34},
    {   -29,     63,     50,    -23,    137,     25,     64,     79,     32,     81,     54,     75,     62,      5,     35,      8,     -9,    -28,      2,    -52,     -2,      7,     -8,    -12,    -19,     -4,    -30,    -17,    -25,    -64},
    {    85,     32,     11,     48,     90,     79,     16,     35,    -25,     21,     29,    -23,      7,     86,      6,     28,    -29,    -24,    -16,      9,     56,    -28,      4,    -14,     -6,    -49,      6,    -21,    -36,      9},
    {    51,    -13,    -16,    -97,    -39,     74,    -82,    -71,    -45,     68,      8,    -42,    -40,    -49,    170,    -82,     -2,     -3,     -4,     35,   -140,   -126,      0,    -11,     78,    -31,     17,    -25,     -4,     79},
};

inline const int V3[7] = {0, 12, 29, 17, 13, 33, 28};

constexpr int v4_mobility_weight = 0;

// V5: 百手結算 (Tanh 判定)
constexpr int v5_time_scale = 1646;
constexpr double v5_tanh_c = 0.5;
constexpr double v5_expo = 8.2986;

inline const int V6[30] = {    15,     13,     14,     31,     14,     12,     16,     20,     17,     10,      9,     18,     11,     14,      0,     18,     16,      9,     11,      8,     23,     21,     12,     11,      4,     16,      9,     12,     10,      6};
constexpr double v6_expo = 0.7119;

constexpr int v_intercept = 3;
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