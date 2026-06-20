#include <algorithm>
#include <chrono>
#include <utility>

#include "state.hpp"
#include "duckyQuack.hpp"

namespace DuckyQuack {

// *=============================================*
// State，嗯笑死我沒有要用你寫的唷，而且我還塞在同一份檔案
static int custom_evaluate(State* state, bool use_mobility = true) {
    if (state->game_state == WIN) return P_MAX;

    auto self_board = state->board.board[state->player];
    auto oppn_board = state->board.board[1 - state->player];
    int self_score = 0, oppn_score = 0;

    int self_kr = -1, self_kc = -1;
    int oppn_kr = -1, oppn_kc = -1;

    // 1. 快速尋找雙方國王 (供 V2 曼哈頓距離使用)
    for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
            if (self_board[r][c] == 6) { self_kr = r; self_kc = c; }
            if (oppn_board[r][c] == 6) { oppn_kr = r; oppn_kc = c; }
        }
    }

    auto manhattan = [](int r1, int c1, int r2, int c2) {
        return std::abs(r1 - r2) + std::abs(c1 - c2);
    };

    // V6 所需：步數保護機制 (防呆，避免異常值導致權重爆炸)
    int safe_step = state->step;
    if (safe_step < 0 || safe_step > 300) safe_step = 20;

    // 2. 靜態區，嚴格對齊公式：V1, V2, V3, V5, V6
    for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
            
            // --- 幫我方計算 ---
            int sp = self_board[r][c];
            if (sp > 0) {
                // 視角正規化
                int pr = (state->player == 0) ? r : (BOARD_H - 1 - r);
                int idx = pr * BOARD_W + c;

                // [公式 V1]: 各種棋子數量差 (獨立計算 Piece Value)
                self_score += PIECE_VALUES[sp];
                
                // [公式 V3]: 各種棋子的位置 (純 PST)
                self_score += u3[sp][idx];

                // [公式 V2]: 與敵方國王的曼哈頓距離 (國王 sp=6 本身不計算追殺)
                if (sp < 6 && oppn_kr != -1) {
                    int dist = manhattan(r, c, oppn_kr, oppn_kc);
                    self_score += u2[sp - 1] * dist; 
                }

                // [公式 V5]: 漸進式升變指數 (剩 1~4 步)
                if (sp == 1 && pr >= 1 && pr <= 4) {
                    self_score += u5_pawn_stages[pr - 1];
                }

                // [公式 V6]: 當前步數 * 國王位置權重 (御駕親征)
                if (sp == 6) {
                    self_score += u7_king_step[idx] * safe_step;
                }
            }
            
            // --- 幫敵方計算 (維持對稱性) ---
            int op = oppn_board[r][c];
            if (op > 0) {
                // 視角正規化 (敵方的底線是他的 0)
                int opr = (state->player == 1) ? r : (BOARD_H - 1 - r);
                int idx = opr * BOARD_W + c;
                
                // [公式 V1]
                oppn_score += PIECE_VALUES[op];

                // [公式 V3]
                oppn_score += u3[op][idx];

                // [公式 V2]
                if (op < 6 && self_kr != -1) {
                    int dist = manhattan(r, c, self_kr, self_kc);
                    oppn_score += u2[op - 1] * dist;
                }

                // [公式 V5]
                if (op == 1 && opr >= 1 && opr <= 4) {
                    oppn_score += u5_pawn_stages[opr - 1];
                }

                // [公式 V6]
                if (op == 6) {
                    oppn_score += u7_king_step[idx] * safe_step;
                }
            }
        }
    }

    int dynamic = 0;

    // 3. 動態探索區：對齊公式中的 V4 與 k1
    if (use_mobility) {
        // 確保己方合法步生成完畢 (防呆)
        if (state->legal_actions.empty() && state->game_state == UNKNOWN) {
            state->get_legal_actions();
        }

        int self_mobility = state->legal_actions.size();
        int self_tactical = 0;
        
        // 掃描我方可以吃誰
        for (const auto& action : state->legal_actions) {
            int tr = action.second.first % BOARD_H;
            int tc = action.second.second;
            int victim = oppn_board[tr][tc]; // 我方目標格上的敵子
            if (victim > 0) {
                self_tactical += u4_tactical[victim - 1];
            }
        }
        
        // 生成敵方的盤面來算他們的機動力與戰術威脅
        State opp_state(state->board, 1 - state->player);
        opp_state.get_legal_actions();
        int oppn_mobility = opp_state.legal_actions.size();
        int oppn_tactical = 0;
        
        // 掃描敵方可以吃誰
        for (const auto& action : opp_state.legal_actions) {
            int tr = action.second.first % BOARD_H;
            int tc = action.second.second;
            int victim = self_board[tr][tc];
            if (victim > 0) {
                oppn_tactical += u4_tactical[victim - 1]; 
            }
        }

        // [公式 V4]: 下一手吃子的步數權重差
        dynamic += (self_tactical - oppn_tactical);
        // [公式 k1]: 合法步數量差 (機動力)
        dynamic += u6_mobility * (self_mobility - oppn_mobility);
    }

    // 結算：基礎常數 + 靜態雙方差距 + 動態威脅差距
    return u_intercept + (self_score - oppn_score) + dynamic;
}

// ==============================================================================
// TT
enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWERBOUND = 1, TT_UPPERBOUND = 2 };

struct TTEntry {
    uint64_t hash = 0;       
    int score = 0;           
    Move best_move = {{-1, -1}, {-1, -1}}; 
    int8_t depth = -1; // 這邊鑽到了第幾層
    TTFlag flag = TT_EXACT; // 預設都是 EXACT 啦
    bool valid = false;  
};

static TTEntry* tt_table = new TTEntry[TT_SIZE]();  // 不敢用 vector 了對不起

// 用來把 hash 轉回去 index 
static inline size_t tt_index(uint64_t hash){ return hash & (TT_SIZE - 1); } 
// 用來把 hash 轉回去 index
static const TTEntry* tt_probe(uint64_t hash){
    const TTEntry& tmp = tt_table[tt_index(hash)];
    if(tmp.valid && tmp.hash == hash) return &tmp;
    return nullptr;
}

static void tt_store(uint64_t hash, int score, int depth, TTFlag flag, const Move& best){
    TTEntry& e = tt_table[tt_index(hash)];
    e.hash      = hash;
    e.score     = score;
    e.depth     = (int8_t) depth;
    e.flag      = flag;
    e.best_move = best;
    e.valid     = true;
}

// 修正 Ply-Drift：將絕殺分數轉換為絕對距離與相對距離
static inline int score_to_tt(int score, int ply) {
    // 轉換成絕對距離 (把當前的 ply 補回去)
    if (score >= P_MAX - MAX_PLY) return score + ply;
    if (score <= M_MAX + MAX_PLY) return score - ply;
    return score;
}

// 修正 Ply-Drift：將絕殺分數轉換為絕對距離與相對距離
static inline int score_from_tt(int score, int ply) {
    // 讀取時，將絕對距離扣掉現在的 ply，轉回相對距離
    if (score >= P_MAX - MAX_PLY) return score - ply;
    if (score <= M_MAX + MAX_PLY) return score + ply;
    return score;
}

// ==============================================================================
// Killer Heuristic + MVV-LVA

static Move killer_moves[MAX_PLY][NUM_KILLERS];

static int score_move(State* state, const Move& move, const Move& tt_move, int ply) {
    // 如果 TT 有就先用 TT 的
    if (move == tt_move) return TT_MOVE_SCORE;

    int opp = 1 - state->player;
    int p = state->player;
    int fr = move.first.first, fc = move.first.second;  // from row, column
    int tr = move.second.first % BOARD_H, tc = move.second.second; // to row, column

    int attacker = state->board.board[p][fr][fc];
    int victim = state->board.board[opp][tr][tc];

    // 第二順位，吃子步
    if (victim != 0) {
        return CAPTURE_BASE + (MVV_VALUES[victim] * 10) - MVV_VALUES[attacker];
    }

    // 第三順位，升變步
    if(attacker == 1){
        bool promotes = (state->player == 0 && tr == 0) || (state->player == 1 && tr == BOARD_H-1);
        if(promotes) return PROMOTE_BASE;
    }

    // 第四順位，殺手步
    if (ply >= 0 && ply < MAX_PLY) {
        for (int i = 0; i < NUM_KILLERS; ++i) {
            if (move == killer_moves[ply][i]) {
                return KILLER_BASE - (i * KILLER_PENALTY); 
            }
        }
    }

    // 最後一個順位，安靜步
    return NORMAL_MOVE;
}

// 拿來排序各步數的
static void order_moves(std::vector<Move>& moves, State* state, const Move& tt_move, int ply){
    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return score_move(state, a, tt_move, ply) > score_move(state, b, tt_move, ply);
        });
}

// ==============================================================================
// QS

int Policy::q_search(State *state, GameHistory& history, int ply, SearchContext& ctx, const MMParams& p, int alpha, int beta) {
    ctx.nodes++;
    if(ctx.stop) return 0; // 就是說，好了就好了辣
    if(ply > ctx.seldepth) ctx.seldepth = ply;

    // 先看我如果連動都不動會發生什麼事
    int stand_pat = custom_evaluate(state, false); // QS 別算機動力了，省時
    if(stand_pat >= beta) return beta;  // 那你走不過來，可以剪了
    if(stand_pat > alpha) alpha = stand_pat;  // 更新新的 alpha

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }
    
    if(state->game_state == WIN) return P_MAX - ply; 
    if(state->game_state == DRAW) return 0;

    int opp = 1 - state->player;
    std::vector<Move> wt;
    wt.reserve(12);

    // 我們只看吃子步與升變步
    for(auto& action : state->legal_actions){
        int fr = action.first.first, fc = action.first.second;
        int tr = action.second.first % BOARD_H, tc = action.second.second;
        
        int captured = state->board.board[opp][tr][tc];
        int attacker = state->board.board[state->player][fr][fc];
        bool is_promotion = (attacker == 1) && ((state->player == 0 && tr == 0) || (state->player == 1 && tr == BOARD_H - 1));
        
        if(captured || is_promotion) wt.push_back(action);
    }

    Move dummy = {{-1, -1}, {-1, -1}};
    order_moves(wt, state, dummy, -1);

    for(auto& action : wt){
        if(ctx.stop) break; 
        State* next = state->next_state(action);
        // 這根本不該發生說真的
        if(next->legal_actions.empty() && next->game_state == UNKNOWN) next->get_legal_actions();

        int score;
        score = -q_search(next, history, ply + 1, ctx, p, -beta, -alpha);
        delete next;

        if(score >= beta) return beta;
        if(score > alpha) alpha = score; // 更新 alpha
    }
    return alpha;
}

// ==============================================================================
// PVS + MVV-LVA + Killer Heuristic

int Policy::eval_ctx(State *state, int depth, GameHistory& history, int ply, SearchContext& ctx, const MMParams& p, int alpha, int beta) {  
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0; // 誒記得跳出去

    // 結束了？結束了。
    if(state->legal_actions.empty() && state->game_state == UNKNOWN) state->get_legal_actions();
    if(state->game_state == WIN) return P_MAX - ply; 
    if(state->game_state == DRAW) return 0;

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    uint64_t ha = state->hash();

    Move tt_best = {{-1, -1}, {-1, -1}};

    // 如果 TT 找得到，就先用 TT 的
    const TTEntry* tte = tt_probe(ha);
    if(tte && tte->depth >= depth){ // 直接用就好
        int s = score_from_tt(tte->score, ply);
        if(tte->flag == TT_EXACT) return s;
        if(tte->flag == TT_LOWERBOUND && s >= beta) return s; 
        if(tte->flag == TT_UPPERBOUND && s <= alpha) return s;
        tt_best = tte->best_move;
    } else if(tte){ // 不夠深？那至少我們猜測它是好的嘛
        tt_best = tte->best_move;
    }

    // 葉節點，丟給 QS 處理
    if(depth <= 0) return q_search(state, history, ply, ctx, p, alpha, beta);
    history.push(ha);

    // MVV-LVA：等等要排序的東西
    std::vector<Move> moves = state->legal_actions;
    order_moves(moves, state, tt_best, ply);

    int best_score = -INF_MAX;
    Move best_move = (moves.empty()) ? (Move{{-1, -1}, {-1, -1}}) : (moves[0]);
    int original_alpha = alpha;
    bool first_step = true;

    for(auto& action : moves){
        if(ctx.stop) break;

        State* next = state->next_state(action); 
        int score;

        if(first_step){
            // PVS：第一步用完整視窗算
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            score = -score;
            first_step = false;
        } else {
            // PVS：其他步用超窄視窗快速看一下就好
            int null_alpha = -(alpha + 1);
            int null_beta  = -alpha;
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, null_alpha, null_beta);
            score = -score;

            // 翻車了，那你就乖乖用完整視窗重算吧
            if(score > alpha && score < beta){ 
                score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
                score = -score;
            }
        }
        delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
        }
        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){
            // Killer Heuristic：別急著跳出去，我們先存殺手步
            int opp = 1 - state->player;
            int tr = action.second.first % BOARD_H, tc = action.second.second;
            if (state->board.board[opp][tr][tc] == 0 && ply < MAX_PLY) {
                bool exists = false;
                for (int i = 0; i < NUM_KILLERS; ++i) {
                    if (killer_moves[ply][i] == action) { exists = true; break; }
                }
                if (!exists) {
                    for (int i = NUM_KILLERS - 1; i > 0; --i) killer_moves[ply][i] = killer_moves[ply][i - 1];
                    killer_moves[ply][0] = action;
                }
            }
            break; // Beta 剪枝
        }
    }

    history.pop(ha);

    if(!ctx.stop){
        TTFlag flag;
        if(best_score <= original_alpha) flag = TT_UPPERBOUND; 
        else if(best_score >= beta)      flag = TT_LOWERBOUND; 
        else                             flag = TT_EXACT; 
        tt_store(ha, score_to_tt(best_score, ply), depth, flag, best_move);
    }

    return best_score;
}

// ==============================================================================
// 從根結點發起搜尋

SearchResult Policy::search(State *state, int depth, GameHistory& history, SearchContext& ctx){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    // 每層的殺手步不太能沿用，這個還是要重製
    for(int i = 0; i < MAX_PLY; ++i){
        for(int j = 0; j < NUM_KILLERS; ++j) killer_moves[i][j] = {{-1, -1}, {-1, -1}};
    }

    if(!state->legal_actions.size()) state->get_legal_actions();
    if(state->legal_actions.empty()) return result; 

    result.best_move = state->legal_actions[0];
    result.score = -INF_MAX;

    for(int d = 1; d <= depth; d++){
        if(ctx.stop) break; 

        ctx.seldepth = 0;
        int current_best_score = -INF_MAX; 
        Move current_best_move = state->legal_actions[0];

        std::vector<Move> root_moves = state->legal_actions;
        Move tt_best = {{-1, -1}, {-1, -1}};

        // 如果 TT 有能用的，我們就先猜測裡面是最好的
        const TTEntry* tte = tt_probe(state->hash());
        if(tte) tt_best = tte->best_move; 
        order_moves(root_moves, state, tt_best, 0);

        int move_index = 0;
        int total_moves = (int)root_moves.size();
        bool aborted = false; // 是被中途截斷的嗎？是的話不能更新進 result
        bool is_first_move = true; 

        for(auto& action : root_moves){
            if(ctx.stop){ aborted = true; break; }

            State* next = state->next_state(action);
            int score;

            if(is_first_move){ // PVS：我懶得打了
                score = eval_ctx(next, d - 1, history, 1, ctx, p, -INF_MAX, INF_MAX);
                score = -score;
                is_first_move = false;
            } else { // PVS：我懶得打了
                int na = -(current_best_score + 1);
                int nb = -current_best_score;
                score = eval_ctx(next, d - 1, history, 1, ctx, p, na, nb);
                score = -score;
                
                if(score > current_best_score && !ctx.stop){
                    score = eval_ctx(next, d - 1, history, 1, ctx, p, -INF_MAX, -current_best_score);
                    score = -score;
                }
            }
            delete next;

            if(!ctx.stop && !aborted && score > current_best_score){
                current_best_score = score;
                current_best_move = action;

                if(p.report_partial && ctx.on_root_update){
                    ctx.on_root_update({current_best_move, current_best_score, d, move_index + 1, total_moves});
                }
            }
            move_index++;
        }
        
        // 只有完整算完的層數才能更新進 result！否則當作我在擴充 TT 就好
        if(!aborted && !ctx.stop){
            result.best_move = current_best_move;
            result.score = current_best_score;
            result.depth = d;
            result.nodes = ctx.nodes;
            result.pv = {current_best_move};

            // 你贏了嗎？如果贏了就不用算了
            if(std::abs(result.score) >= P_MAX - 100) break;
        }
    }

    result.nodes = ctx.nodes; 
    return result;
}

ParamMap Policy::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "true"},
    };
}

std::vector<ParamDef> Policy::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
    };
}

} // namespace DuckyQuack