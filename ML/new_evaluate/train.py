import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

INPUT_CSV = "../extracted/dataset.csv"
EPOCHS = 15                  
BATCH_SIZE = 1024            
LEARNING_RATE = 0.05         
WEIGHT_DECAY = 1e-4          

def get_pseudo_legal_and_tactical(board, player):
    sign = 1 if player == 'w' else -1
    mobility = 0
    tac_counts = np.zeros(7) 
    
    dirs = {
        2: [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)], 
        3: [(-1,-1), (-1,1), (1,-1), (1,1)], 
        4: [(-1,0), (1,0), (0,-1), (0,1)], 
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
                                tac_counts[abs(target)] += 1
                elif ptype in [2, 6]: 
                    for dr, dc in dirs[ptype]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 5:
                            target = board[nr][nc]
                            if target == 0:
                                mobility += 1
                            elif target * sign < 0:
                                tac_counts[abs(target)] += 1
                else: 
                    for dr, dc in dirs[ptype]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 6 and 0 <= nc < 5:
                            target = board[nr][nc]
                            if target == 0:
                                mobility += 1
                            else:
                                if target * sign < 0:
                                    tac_counts[abs(target)] += 1
                                break 
                            nr += dr
                            nc += dc
    return mobility, tac_counts

class MiniChessDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset: {csv_file}")
        df = pd.read_csv(csv_file)
        
        self.features, self.labels = [], []
        # 僅供 V5 判定使用的官方數值
        official_vals = {1:2, 2:7, 3:8, 4:6, 5:20, 6:0}
        
        for _, row in df.iterrows():
            result = row['result']
            step = row['step']
            
            prob = 0.5 if result == 0.5 else (1.0 if result == 1.0 else 0.0)
            self.labels.append([prob])
            
            board = np.zeros((6, 5), dtype=int)
            for i in range(30):
                board[i//5][i%5] = row[f'sq_{i}']
                
            v1_diff = np.zeros(6)        
            v2_pst = np.zeros(6 * 30)    
            v6_king = np.zeros(30)       
            self_off = 0
            oppn_off = 0                   
            
            for r in range(6):
                for c in range(5):
                    val = board[r][c]
                    if val == 0: continue
                    
                    ptype = abs(val) 
                    sign = 1 if val > 0 else -1
                    
                    v1_diff[ptype - 1] += sign
                    
                    if sign == 1:
                        self_off += official_vals[ptype]
                        idx = r * 5 + c
                    else:
                        oppn_off += official_vals[ptype]
                        idx = (5 - r) * 5 + c
                        
                    pst_feature_idx = (ptype - 1) * 30 + idx
                    v2_pst[pst_feature_idx] += sign
                    
                    if ptype == 6:
                        v6_king[idx] += sign 
            
            w_mob, w_tac = get_pseudo_legal_and_tactical(board, 'w')
            b_mob, b_tac = get_pseudo_legal_and_tactical(board, 'b')
            
            v3_tac = w_tac[1:] - b_tac[1:] 
            mob_diff = w_mob - b_mob
            off_diff = self_off - oppn_off
            
            # 拼接: V1(6) + V2(180) + V3(6) + Mob(1) + Off_Diff(1) + V6(30) + Step(1) = 225維
            self.features.append(np.concatenate([v1_diff, v2_pst, v3_tac, [mob_diff], [off_diff], v6_king, [step]]))
            
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class MiniChessEvaluator(nn.Module):
    def __init__(self):
        super(MiniChessEvaluator, self).__init__()
        # ==========================================
        # V1: 解除所有束縛，讓神經網路自由通靈戰鬥價值
        # ==========================================
        self.v1 = nn.Parameter(torch.ones(6))
        
        self.v2 = nn.Parameter(torch.zeros(180))
        self.raw_v3 = nn.Parameter(torch.ones(6) * 0.1)
        
        # 強迫把機動力設為零，且不參與訓練 (防止水平效應)
        self.v4_weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # V5 百手結算參數
        self.v5_time_scale = 15.0  
        self.v5_tanh_c = 0.5       
        self.v5_expo = nn.Parameter(torch.tensor(2.0))
        
        self.v6 = nn.Parameter(torch.ones(30))
        self.v6_expo = nn.Parameter(torch.tensor(1.0))
        self.v_intercept = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_v1 = x[:, 0:6]
        x_v2 = x[:, 6:186]
        x_v3 = x[:, 186:192]
        x_mob_diff = x[:, 192]
        x_off_diff = x[:, 193]  
        x_v6_king = x[:, 194:224]
        x_step = x[:, 224]
        
        safe_step = torch.clamp(x_step, min=0.0, max=300.0)
        safe_step = torch.where(safe_step == 0, torch.tensor(20.0, device=x.device), safe_step)
        
        v3 = nn.functional.softplus(self.raw_v3)
        v5_e = nn.functional.softplus(self.v5_expo)
        v6_e = nn.functional.softplus(self.v6_expo)
        
        # V1 直接相乘，無條件信任模型給出的權重
        score_v1 = x_v1 @ self.v1
        score_v2 = x_v2 @ self.v2
        score_v3 = x_v3 @ v3
        score_v4 = x_mob_diff * self.v4_weight
        
        time_factor = torch.pow(safe_step / 100.0, v5_e)
        score_v5 = self.v5_time_scale * torch.tanh(self.v5_tanh_c * x_off_diff) * time_factor
        
        score_v6 = (x_v6_king @ self.v6) * torch.pow(safe_step, v6_e)
        
        return self.v_intercept + score_v1 + score_v2 + score_v3 + score_v4 + score_v5 + score_v6

def train():
    dataset = MiniChessDataset(INPUT_CSV)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MiniChessEvaluator()
    # 過濾 requires_grad=False 的參數
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        total_loss, correct_preds, total_samples, mae_sum = 0.0, 0, 0, 0.0
        
        for features, labels in dataloader:
            optimizer.zero_grad()
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            preds = (probs >= 0.5).float()
            correct_preds += (preds.squeeze() == (labels.squeeze() > 0.5)).sum().item()
            
            mae_sum += torch.abs(probs.unsqueeze(1) - labels).sum().item() 
            total_samples += labels.size(0)
            
            loss = criterion(logits.unsqueeze(1), labels)
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
    v4_w = model.v4_weight.item()
    
    v5_e = nn.functional.softplus(model.v5_expo).item()
    
    raw_v6 = model.v6.detach().numpy().copy()
    v6_e = nn.functional.softplus(model.v6_expo).item()
    intercept = model.v_intercept.item()
    
    # 計算安全系數時使用 raw_v1
    max_v1 = np.max(np.abs(raw_v1)) * 6 + np.sum(np.abs(raw_v1))
    max_v2 = np.max(np.abs(raw_v2)) * 10
    max_v3 = np.max(np.abs(v3)) * 10
    max_v4 = abs(v4_w) * 20
    max_v5 = model.v5_time_scale * (3.0 ** v5_e)
    max_v6 = np.max(np.abs(raw_v6)) * (300.0 ** v6_e)
    
    max_raw_score = abs(intercept) + max_v1 + max_v2 + max_v3 + max_v4 + max_v5 + max_v6
    
    SAFE_LIMIT = 15000000.0
    output_scale = SAFE_LIMIT / max_raw_score if max_raw_score > 0 else 1000.0
    if output_scale > 100000.0: output_scale = 100000.0
    
    v1 = (raw_v1 * output_scale).astype(int)
    v2 = (raw_v2 * output_scale).astype(int)
    v3 = (v3 * output_scale).astype(int)
    v6 = (raw_v6 * output_scale).astype(int)
    
    v4_w_scaled = int(v4_w * output_scale)
    v5_time_scale_scaled = int(model.v5_time_scale * output_scale)
    intercept_scaled = int(intercept * output_scale)
    
    print(f"// OUTPUT_SCALE = {output_scale:.2f}")
    print(f"// Max Theoretical Eval = {int(max_raw_score * output_scale):,d}")
    
    print(f"inline const int V1_OFFICIAL[7] = {{0, 2, 7, 8, 6, 20, 0}};")
    print(f"inline const int V1[7] = {{0, {v1[0]}, {v1[1]}, {v1[2]}, {v1[3]}, {v1[4]}, {v1[5]}}};\n")
    
    print("// V2: PST 陣型評估")
    print("inline const int V2[7][30] = {")
    print("    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},")
    for i in range(6):
        row_vals = v2[i*30 : (i+1)*30]
        row_str = ", ".join(f"{val:6d}" for val in row_vals)
        print(f"    {{{row_str}}},")
    print("};\n")
    
    print(f"inline const int V3[7] = {{0, {v3[0]}, {v3[1]}, {v3[2]}, {v3[3]}, {v3[4]}, {v3[5]}}};\n")
    print(f"constexpr int v4_mobility_weight = {v4_w_scaled};\n")
    
    print("// V5: 百手結算 (Tanh 判定)")
    print(f"constexpr int v5_time_scale = {v5_time_scale_scaled};")
    print(f"constexpr double v5_tanh_c = {model.v5_tanh_c};")
    print(f"constexpr double v5_expo = {v5_e:.4f};\n")
    
    v6_str = ", ".join(f"{val:6d}" for val in v6)
    print(f"inline const int V6[30] = {{{v6_str}}};")
    print(f"constexpr double v6_expo = {v6_e:.4f};\n")
    print(f"constexpr int v_intercept = {intercept_scaled};")

if __name__ == "__main__":
    train()