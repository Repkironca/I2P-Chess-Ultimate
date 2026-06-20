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
constexpr int TT_SIZE = 1 << 22;

// ==============================================================================
// 基礎棋力常數
// ==============================================================================
// 空格, 小兵, 騎士, 主教, 城堡, 皇后, 國王
inline const int PIECE_VALUES[7] = {0, 20, 60, 70, 80, 200, 1000};
inline const int MVV_VALUES[7]   = {0, 10, 50, 30, 30, 90,  900};

// ==============================================================================
// ML 特徵向量
// ==============================================================================
inline const int u2[5] = {1, 1, 1, 1, 1}; // 曼哈頓權重

inline const int u3[7][30] = { // PST 權重，{空格, 小兵, 騎士, 主教, 城堡, 皇后, 國王}
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
};

inline const int u4_tactical[6] = {1, 1, 1, 1, 1, 1}; // 吃子威脅權重
inline const int u5_pawn_stages[4] = {1, 1, 1, 1}; // 升變懲罰
constexpr int u6_mobility = 1; // 機動力常數

// 國王親征權重
inline const int u7_king_step[30] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
constexpr int u_intercept = 1;

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