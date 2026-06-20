import pandas as pd
import numpy as np
import os

def init_board():
    grid = [[0]*5 for _ in range(6)]
    # Black (正數)
    grid[0] = [6, 5, 4, 3, 2] # K, Q, B, N, R
    grid[1] = [1, 1, 1, 1, 1] # P
    # White (負數)
    grid[4] = [-1, -1, -1, -1, -1] # P
    grid[5] = [-2, -3, -4, -5, -6] # R, N, B, Q, K
    return grid

def parse_sq(sq_str):
    col = ord(sq_str[0]) - ord('a')
    row = 6 - int(sq_str[1:])
    return row, col

def get_pseudo_legal_and_tactical(grid, player):
    mobility = 0
    tactical_threats = [0]*6
    
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
            
            if ptype == 1:
                dir = -1 if player == 0 else 1
                if 0 <= r + dir < 6:
                    if grid[r + dir][c] == 0:
                        targets.append((r + dir, c))
                    if c > 0 and grid[r + dir][c - 1] != 0 and (grid[r + dir][c - 1] > 0) != (player == 1):
                        targets.append((r + dir, c - 1))
                    if c < 4 and grid[r + dir][c + 1] != 0 and (grid[r + dir][c + 1] > 0) != (player == 1):
                        targets.append((r + dir, c + 1))
            elif ptype == 3:
                for i in range(8):
                    nr, nc = r + knight_dr[i], c + knight_dc[i]
                    if 0 <= nr < 6 and 0 <= nc < 5:
                        targets.append((nr, nc))
            elif ptype == 6:
                for i in range(8):
                    nr, nc = r + dr[i], c + dc[i]
                    if 0 <= nr < 6 and 0 <= nc < 5:
                        targets.append((nr, nc))
            else:
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
                elif (target_p > 0) != (player == 1):
                    mobility += 1
                    tactical_threats[abs(target_p) - 1] += 1
                    
    return mobility, tactical_threats

def extract_features():
    # 宣告檔案來源與其基礎權重
    datasets = [('dataset_hq.csv', 1.0)]
    if os.path.exists('dataset_ultra.csv'):
        print("偵測到 dataset_ultra.csv，套用 10 倍權重加成！")
        datasets.append(('dataset_ultra.csv', 10.0))
        
    X_list, y_list, step_list, weight_list = [], [], [], []
    
    for file_path, base_w in datasets:
        try:
            df = pd.read_csv(file_path)
            print(f"處理 {file_path} 中... ({len(df)} 場對局)")
        except FileNotFoundError:
            continue
            
        for idx, row in df.iterrows():
            res_str = str(row['Result'])
            if '1.0' in res_str or '1' == res_str: y_val = 1
            elif '0.0' in res_str or '0' == res_str: y_val = -1
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
                
                f = np.zeros(230)
                w_kr, w_kc, b_kr, b_kc = -1, -1, -1, -1
                for r in range(6):
                    for c in range(5):
                        if grid[r][c] == -6: w_kr, w_kc = r, c
                        if grid[r][c] == 6: b_kr, b_kc = r, c
                        
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
                            f[ptype - 1] += sign 
                            
                            opp_kr = b_kr if is_white else w_kr
                            opp_kc = b_kc if is_white else w_kc
                            if opp_kr != -1:
                                dist = abs(r - opp_kr) + abs(c - opp_kc)
                                f[5 + ptype - 1] += sign * dist
                                
                        f[10 + (ptype - 1) * 30 + idx_pst] += sign
                        
                        if ptype == 1 and 1 <= pr <= 4:
                            f[190 + pr - 1] += sign
                            
                        if ptype == 6:
                            f[194 + idx_pst] += sign * safe_step
                            
                w_mob, w_tac = get_pseudo_legal_and_tactical(grid, 0)
                b_mob, b_tac = get_pseudo_legal_and_tactical(grid, 1)
                
                for i in range(5):
                    f[224 + i] = w_tac[i] - b_tac[i]
                    
                f[229] = w_mob - b_mob
                
                X_list.append(f)
                y_list.append(y_val)
                step_list.append(step)
                weight_list.append(base_w)
                
                player = 1 - player
                step += 1
                
    np.savez_compressed('features.npz', X=np.array(X_list), y=np.array(y_list), steps=np.array(step_list), weights=np.array(weight_list))
    print("特徵萃取完成，已儲存至 features.npz")

if __name__ == '__main__':
    extract_features()