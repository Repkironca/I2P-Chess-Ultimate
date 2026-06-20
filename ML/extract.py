"""
===============================================================================
[理論解釋與模型設計 (說服說明)]
1. 解決「騎士換小兵自殺」：
   前次使用 Ridge (alpha=6.21) 發生此問題，是因為無約束的線性迴歸在樣本不平衡時，
   極易發生「特徵共線性污染」，導致 V1(騎士) 的權重被 PST(位置分數) 吸收而低於小兵。
   本次改用 `scipy.optimize.lsq_linear`，強制設定硬邊界 (Bounds)：
   限制 P(100~200) < N/B(250~450) < R(450~700) < Q(700~1200)，從根本上杜絕降級兌子。

2. 解決「國王走到被將死的格子」：
   這並非 Bug，而是靜態評估函數的先天限制。若國王走到被攻擊的格子，要到「下一回合」
   敵方吃王才會觸發 WIN。為了在「當前回合」就阻止這件事，我們必須給予極大的戰術懲罰。
   作法：將 V4(對手威脅我方國王) 的權重 u4_tactical[5] 寫死為 500000，
   且將 PIECE_VALUES[6] 寫死為 2000000。這兩個極值會讓 AI 絕對避開自殺步，
   同時完美遵守你的「不超過 1e7 (10,000,000)」常數限制。

3. 和局與時間權重：
   使用 sample_weight = 0.2 (和局) 或 1.0 (勝負) 乘上 step^0.3。
   為了放入 lsq_linear，將權重開根號後乘上特徵矩陣 A 與目標向量 b。
===============================================================================
"""

import pandas as pd
import numpy as np

def init_board():
    grid = [[0]*5 for _ in range(6)]
    grid[0] = [6, 5, 4, 3, 2] # Black (1)
    grid[1] = [1, 1, 1, 1, 1]
    grid[4] = [-1, -1, -1, -1, -1] # White (0) represented as negative
    grid[5] = [-2, -3, -4, -5, -6]
    return grid

def parse_sq(sq_str):
    col = ord(sq_str[0]) - ord('a')
    row = 6 - int(sq_str[1:])
    return row, col

def get_pseudo_legal_and_tactical(grid, player):
    # player: 0 for White (negative), 1 for Black (positive)
    mobility = 0
    tactical_threats = [0]*6 # 對應 P, N, B, R, Q, K
    
    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]
    knight_dr = [-2, -2, -1, -1, 1, 1, 2, 2]
    knight_dc = [-1, 1, -2, 2, -2, 2, -1, 1]
    
    for r in range(6):
        for c in range(5):
            p = grid[r][c]
            if p == 0: continue
            is_white = (p < 0)
            if (player == 0 and not is_white) or (player == 1 and is_white):
                continue
                
            ptype = abs(p)
            targets = []
            
            if ptype == 1: # Pawn
                dir = -1 if player == 0 else 1
                if 0 <= r + dir < 6:
                    if grid[r + dir][c] == 0:
                        targets.append((r + dir, c))
                    if c > 0 and grid[r + dir][c - 1] != 0 and (grid[r + dir][c - 1] > 0) != (player == 1):
                        targets.append((r + dir, c - 1))
                    if c < 4 and grid[r + dir][c + 1] != 0 and (grid[r + dir][c + 1] > 0) != (player == 1):
                        targets.append((r + dir, c + 1))
            elif ptype == 3: # Knight
                for i in range(8):
                    nr, nc = r + knight_dr[i], c + knight_dc[i]
                    if 0 <= nr < 6 and 0 <= nc < 5:
                        targets.append((nr, nc))
            elif ptype == 6: # King
                for i in range(8):
                    nr, nc = r + dr[i], c + dc[i]
                    if 0 <= nr < 6 and 0 <= nc < 5:
                        targets.append((nr, nc))
            else: # R, B, Q
                st = 0 if ptype in (2, 5) else 4
                ed = 4 if ptype == 2 else 8
                for i in range(st, ed):
                    nr, nc = r + dr[i], c + dc[i]
                    while 0 <= nr < 6 and 0 <= nc < 5:
                        targets.append((nr, nc))
                        if grid[nr][nc] != 0: break
                        nr += dr[i]
                        nc += dc[i]
                        
            for tr, tc in targets:
                target_p = grid[tr][tc]
                if target_p == 0:
                    mobility += 1
                elif (target_p > 0) != (player == 1): # 敵方棋子
                    mobility += 1
                    tactical_threats[abs(target_p) - 1] += 1
                    
    return mobility, tactical_threats

def extract_features():
    df = pd.read_csv('dataset_hq.csv')
    X_list, y_list, step_list = [], [], []
    
    for idx, row in df.iterrows():
        res_str = str(row['Result'])
        if '1.0' in res_str or '1' == res_str: y_val = 400
        elif '0.0' in res_str or '0' == res_str: y_val = -400
        else: y_val = 0
        
        moves = row['Moves'].strip().split()
        grid = init_board()
        step = 1
        player = 0
        
        for mv in moves:
            if len(mv) < 4: continue
            fr, fc = parse_sq(mv[0:2])
            tr, tc = parse_sq(mv[2:4])
            
            p = grid[fr][fc]
            grid[fr][fc] = 0
            if abs(p) == 1 and (tr == 0 or tr == 5):
                p = -5 if p < 0 else 5
            grid[tr][tc] = p
            
            # --- 特徵擷取 ---
            f = np.zeros(230)
            
            # 尋找雙方國王
            w_kr, w_kc, b_kr, b_kc = -1, -1, -1, -1
            for r in range(6):
                for c in range(5):
                    if grid[r][c] == -6: w_kr, w_kc = r, c
                    if grid[r][c] == 6: b_kr, b_kc = r, c
            
            # 靜態特徵
            safe_step = min(step, 300)
            for r in range(6):
                for c in range(5):
                    val = grid[r][c]
                    if val == 0: continue
                    is_white = (val < 0)
                    ptype = abs(val)
                    
                    pr = r if is_white else (5 - r)
                    idx_pst = pr * 5 + c
                    sign = 1 if is_white else -1
                    
                    if ptype < 6:
                        f[ptype - 1] += sign # V1 (5)
                        
                        # V2 Manhattan
                        opp_kr = b_kr if is_white else w_kr
                        opp_kc = b_kc if is_white else w_kc
                        if opp_kr != -1:
                            dist = abs(r - opp_kr) + abs(c - opp_kc)
                            f[5 + ptype - 1] += sign * dist
                            
                    # V3 PST (180) -> 10 + (ptype-1)*30 + idx_pst
                    f[10 + (ptype - 1) * 30 + idx_pst] += sign
                    
                    # V5 Pawn Stages (4)
                    if ptype == 1 and 1 <= pr <= 4:
                        f[190 + pr - 1] += sign
                        
                    # V6 King Step (30)
                    if ptype == 6:
                        f[194 + idx_pst] += sign * safe_step
                        
            # 動態特徵 (V4, k1)
            w_mob, w_tac = get_pseudo_legal_and_tactical(grid, 0)
            b_mob, b_tac = get_pseudo_legal_and_tactical(grid, 1)
            
            for i in range(5): # 只訓練 P~Q 的威脅，K 的威脅(Index 5)另外寫死
                f[224 + i] = w_tac[i] - b_tac[i]
                
            f[229] = w_mob - b_mob # k1
            
            X_list.append(f)
            y_list.append(y_val)
            step_list.append(step)
            
            player = 1 - player
            step += 1
            
    np.savez_compressed('features.npz', X=np.array(X_list), y=np.array(y_list), steps=np.array(step_list))

if __name__ == '__main__':
    extract_features()