import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
INPUT_CSV = "../extracted/dataset.csv"
EPOCHS = 15                  
BATCH_SIZE = 1024            
LEARNING_RATE = 0.05         
WEIGHT_DECAY = 1e-4          
# ==========================================

def get_pseudo_legal_and_tactical(board, player):
    sign = 1 if player == 'w' else -1
    mobility = 0
    tac_counts = np.zeros(6) 
    
    dirs = {
        2: [(-1,0), (1,0), (0,-1), (0,1)],
        3: [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)],
        4: [(-1,-1), (-1,1), (1,-1), (1,1)],
        5: [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)],
        6: [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    }
    
    for r in range(6):
        for c in range(5):
            val = board[r][c]
            if val * sign > 0: 
                ptype = abs(val)
                if ptype == 1: 
                    dr = -1 if sign == 1 else 1 
                    nr = r + dr
                    if 0 <= nr < 6 and board[nr][c] == 0:
                        mobility += 1
                    for dc in [-1, 1]:
                        nc = c + dc
                        if 0 <= nr < 6 and 0 <= nc < 5:
                            target = board[nr][nc]
                            if target * sign < 0: 
                                tac_counts[abs(target) - 1] += 1
                elif ptype in [3, 6]: 
                    for dr, dc in dirs[ptype]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 5:
                            target = board[nr][nc]
                            if target == 0:
                                mobility += 1
                            elif target * sign < 0:
                                tac_counts[abs(target) - 1] += 1
                else: 
                    for dr, dc in dirs[ptype]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 6 and 0 <= nc < 5:
                            target = board[nr][nc]
                            if target == 0:
                                mobility += 1
                            else:
                                if target * sign < 0:
                                    tac_counts[abs(target) - 1] += 1
                                break 
                            nr += dr
                            nc += dc
    return mobility, tac_counts

class MiniChessDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset: {csv_file}")
        df = pd.read_csv(csv_file)
        game_max_steps = df.groupby('game_id')['step'].max().to_dict()
        self.features, self.labels, self.weights = [], [], []
        
        for _, row in df.iterrows():
            result = row['result']
            game_id = row['game_id']
            step = row['step']
            max_step = game_max_steps[game_id]
            
            if result == 0.0:
                w, prob = 0.333, 0.5
            else:
                prob = 1.0 if result == 1.0 else 0.0
                w = 1.2 if max_step >= 100 else 1.0
                    
            self.labels.append([prob])
            self.weights.append([w])
            self.features.append(self.parse_fen_to_features(row['board_fen'], step))
            
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        self.weights = torch.tensor(np.array(self.weights), dtype=torch.float32)

    def parse_fen_to_features(self, fen, step):
        piece_map = {'P':1, 'R':2, 'N':3, 'B':4, 'Q':5, 'K':6, 'p':-1, 'r':-2, 'n':-3, 'b':-4, 'q':-5, 'k':-6}
        official_vals = {1:2, 2:6, 3:7, 4:8, 5:20, 6:0}
        
        v1_diff = np.zeros(5)        
        v2_pst = np.zeros(6 * 30)    
        v4_king = np.zeros(30)       
        m_diff = 0                   
        
        board = []
        for row in fen.split('/'):
            r = []
            for char in row:
                if char.isdigit(): r.extend([0] * int(char))
                else: r.append(piece_map[char])
            board.append(r)
            
        for r in range(6):
            for c in range(5):
                val = board[r][c]
                if val == 0: continue
                
                ptype = abs(val) 
                sign = 1 if val > 0 else -1
                
                if ptype <= 5:
                    v1_diff[ptype - 1] += sign
                    m_diff += sign * official_vals[ptype]
                    
                idx = (r * 5 + c) if sign == 1 else ((5 - r) * 5 + c)
                pst_feature_idx = (ptype - 1) * 30 + idx
                
                v2_pst[pst_feature_idx] += sign
                if ptype == 6:
                    v4_king[idx] += sign 
        
        w_mob, w_tac = get_pseudo_legal_and_tactical(board, 'w')
        b_mob, b_tac = get_pseudo_legal_and_tactical(board, 'b')
        
        v3_tac = w_tac - b_tac
        mob_diff = w_mob - b_mob
        
        return np.concatenate([v1_diff, v2_pst, v3_tac, v4_king, [m_diff], [mob_diff], [step]])

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.weights[idx]

class MiniChessEvaluator(nn.Module):
    def __init__(self):
        super(MiniChessEvaluator, self).__init__()
        self.v1 = nn.Parameter(torch.tensor([2.0, 6.0, 7.0, 8.0, 20.0]))
        self.v2 = nn.Parameter(torch.zeros(180))
        
        self.raw_v3 = nn.Parameter(torch.ones(6) * 0.1)
        self.v4 = nn.Parameter(torch.zeros(30))
        
        self.raw_k1 = nn.Parameter(torch.tensor(0.5)) 
        self.raw_v5_w = nn.Parameter(torch.tensor(1.0))
        self.raw_k2 = nn.Parameter(torch.tensor(0.5))
        
        # 【修正】強制 V6 (機動力) 必須為正數
        self.raw_v6_w = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x_v1 = x[:, 0:5]
        x_v2 = x[:, 5:185]
        x_v3 = x[:, 185:191]
        x_v4 = x[:, 191:221]
        x_m_diff = x[:, 221]
        x_mob_diff = x[:, 222]
        x_step = x[:, 223]
        
        v3 = nn.functional.softplus(self.raw_v3)
        k1 = nn.functional.softplus(self.raw_k1)
        v5_w = nn.functional.softplus(self.raw_v5_w)
        k2 = nn.functional.softplus(self.raw_k2) + 1.0 
        
        # 【修正】套用 Softplus 到 V6
        v6_w = nn.functional.softplus(self.raw_v6_w)
        
        score_v1 = x_v1 @ self.v1 
        score_v2 = x_v2 @ self.v2
        score_v3 = x_v3 @ v3
        
        score_v4 = (x_v4 @ self.v4) * torch.pow(torch.clamp(x_step, min=1.0), k1)
        score_v5 = x_m_diff * v5_w * torch.pow(torch.clamp(x_step / 100.0, min=0.01), k2)
        score_v6 = x_mob_diff * v6_w
        
        return score_v1 + score_v2 + score_v3 + score_v4 + score_v5 + score_v6

def train():
    dataset = MiniChessDataset(INPUT_CSV)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MiniChessEvaluator()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for epoch in range(EPOCHS):
        total_loss, correct_preds, total_samples, mae_sum = 0.0, 0, 0, 0.0
        
        for features, labels, weights in dataloader:
            optimizer.zero_grad()
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            preds = (probs >= 0.5).float()
            correct_preds += (preds.squeeze() == (labels.squeeze() > 0.5)).sum().item()
            
            mae_sum += torch.abs(probs.unsqueeze(1) - labels).sum().item() 
            total_samples += labels.size(0)
            
            loss = (criterion(logits.unsqueeze(1), labels) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples * 100
        mae = mae_sum / total_samples
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | MAE: {mae:.4f}")
        
    generate_cpp_code(model)

def generate_cpp_code(model):
    print("\n// ================== C++ Parameters ==================")
    
    raw_v1 = model.v1.detach().numpy().copy()
    raw_v2 = model.v2.detach().numpy().copy()
    
    v3 = nn.functional.softplus(model.raw_v3).detach().numpy().copy()
    raw_v4 = model.v4.detach().numpy().copy()
    
    # 【修正】提取受約束的 v6_w
    v6_w = nn.functional.softplus(model.raw_v6_w).item()
    
    k1 = round(nn.functional.softplus(model.raw_k1).item(), 4)
    v5_w = nn.functional.softplus(model.raw_v5_w).item()
    k2 = round(nn.functional.softplus(model.raw_k2).item() + 1.0, 4)
    
    for i in range(5):
        pst_slice = raw_v2[i*30 : (i+1)*30]
        mean_val = np.mean(pst_slice)
        raw_v1[i] += mean_val             
        raw_v2[i*30 : (i+1)*30] -= mean_val
        
    v4_mean = np.mean(raw_v4)
    raw_v4 -= v4_mean
    raw_v1 = np.maximum(raw_v1, 0.1)
    
    max_v1 = np.max(raw_v1) * 6 + np.sum(raw_v1)
    max_v2 = np.max(np.abs(raw_v2)) * 10
    max_v3 = np.max(np.abs(v3)) * 10
    max_v4 = np.max(np.abs(raw_v4)) * (100 ** k1)
    max_v5 = abs(v5_w) * 51 * 1.0 
    max_v6 = abs(v6_w) * 20
    
    max_raw_score = max_v1 + max_v2 + max_v3 + max_v4 + max_v5 + max_v6
    SAFE_LIMIT = 10000000.0
    output_scale = SAFE_LIMIT / max_raw_score if max_raw_score > 0 else 10000.0
    
    v1 = (raw_v1 * output_scale).astype(int)
    v2 = (raw_v2 * output_scale).astype(int)
    v3 = (v3 * output_scale).astype(int)
    v4 = (raw_v4 * output_scale).astype(int)
    v5_w = int(v5_w * output_scale)
    v6_w = int(v6_w * output_scale)
    
    print(f"// OUTPUT_SCALE = {output_scale:.2f}")
    print(f"// Max Theoretical Eval = {int(max_raw_score * output_scale):,d}")
    print(f"inline const int u1_piece_values[7] = {{0, {v1[0]}, {v1[1]}, {v1[2]}, {v1[3]}, {v1[4]}, 2000000}};")
    
    print("\n// V2: PST 陣型評估 (7個 30維)")
    print("inline const int u2_pst[7][30] = {")
    print("    {0}, // 空格")
    for i in range(6):
        row_vals = v2[i*30 : (i+1)*30]
        row_str = ", ".join(f"{val:6d}" for val in row_vals)
        print(f"    {{{row_str}}},")
    print("};\n")
    
    print("// V3: 吃子威脅權重 (7維)")
    print(f"inline const int u3_tactical[7] = {{0, {v3[0]}, {v3[1]}, {v3[2]}, {v3[3]}, {v3[4]}, {v3[5]}}};\n")
    
    print("// V4: 國王親征特徵 (30維 + K_1)")
    v4_str = ", ".join(f"{val:6d}" for val in v4)
    print(f"inline const int u4_king_outward[30] = {{{v4_str}}};")
    print(f"inline const double u4_k1 = {k1};\n")
    
    print("// V5: 100 手官方結算覺醒 (權重常數 + K_2)")
    print(f"inline const int u5_official_weight = {v5_w};")
    print(f"inline const double u5_k2 = {k2};\n")
    
    print("// V6: 機動力權重")
    print(f"inline const int u6_mobility_weight = {v6_w};\n")
    
    print("constexpr int u_intercept = 0;")
    print("// ==============================================================================")

if __name__ == "__main__":
    train()