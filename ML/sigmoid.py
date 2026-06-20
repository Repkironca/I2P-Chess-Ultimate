import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

def train_model():
    print("載入 features.npz 中...")
    data = np.load('features.npz')
    X = data['X']
    y = data['y']
    steps = data['steps']
    base_weights = data['weights'] # 來自 extract.py 的基礎權重 (1.0 或 10.0)
    
    y_prob = np.where(y > 0, 1.0, np.where(y < 0, 0.0, 0.5))
    
    # 計算樣本權重：和局 0.2，勝負 1.0 -> 乘上 Ultra 加權 -> 乘上步數加成
    w_result = np.where(y == 0, 0.2, 1.0)
    w_sample = w_result * base_weights * (steps ** 0.3)
    w_sample /= np.mean(w_sample)
    
    def loss_fn(w):
        logits = X @ w
        preds = 1 / (1 + np.exp(-np.clip(logits, -10, 10))) 
        bce = -np.mean(w_sample * (y_prob * np.log(preds + 1e-10) + (1 - y_prob) * np.log(1 - preds + 1e-10)))
        l2_reg = 0.005 * np.sum(w**2) 
        return bce + l2_reg

    w0 = np.zeros(230)
    # 初始化 V1: 0=P, 1=R, 2=N, 3=B, 4=Q (嚴格對齊 C++ ptype-1)
    w0[0:5] = [1.0, 5.0, 3.0, 3.0, 9.0] 
    
    bounds = []
    # 【核心修正】強力限制各項權重，確保 Material (V1) > Positional
    # 1. 棋子階級鐵律：P < N,B < R < Q
    bounds.extend([
        (0.8, 1.2),   # Index 0 (Pawn): 基準值約 100
        (4.0, 6.0),   # Index 1 (Rook): 強制在 400~600 區間
        (2.5, 4.0),   # Index 2 (Knight): 強制在 250~400 區間
        (2.5, 4.0),   # Index 3 (Bishop): 強制在 250~400 區間
        (7.0, 12.0)   # Index 4 (Queen): 強制在 700~1200 區間
    ])
    bounds.extend([(-0.15, 0.15)] * 5)    # V2 (曼哈頓): 最高 +- 15cp
    bounds.extend([(-0.3, 0.3)] * 180)    # V3 (PST): 壓死在 +- 30cp 內，嚴禁亂竄
    bounds.extend([(-0.5, 0.5)] * 4)      # V5 (升變階梯): 最高 +- 50cp
    bounds.extend([(-0.02, 0.02)] * 30)   # V6 (國王親征): 極微小調整
    bounds.extend([(0.0, 0.3)] * 5)       # V4 (戰術威脅): 最高 30cp，避免為了威脅而放棄吃子
    bounds.extend([(0.0, 0.1)])           # k1 (機動力): 最高 10cp

    print("開始優化模型 (Logistic Regression / L-BFGS-B)...")
    res = minimize(loss_fn, w0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
    w_opt = res.x
    
    # 縮放魔法 (強制讓小兵等於 100 分)
    scale = 100.0 / w_opt[0]
    coef = np.round(w_opt * scale).astype(int)
    
    preds_class = np.where(X @ w_opt > 0, 1.0, 0.0)
    mask_winloss = (y != 0) 
    acc = accuracy_score(y_prob[mask_winloss], preds_class[mask_winloss], sample_weight=w_sample[mask_winloss])
    
    print(f"\n// Win/Loss Prediction Accuracy (Weighted): {acc:.3f}")
    
    V1 = coef[0:5]
    V2 = coef[5:10]
    V3 = coef[10:190].reshape(6, 30)
    V5 = coef[190:194]
    V6 = coef[194:224]
    V4 = coef[224:229]
    k1 = coef[229]
    
    print("\n// ================== C++ 參數區 (直接複製貼上) ==================")
    # 完美對應: 0, Pawn, Rook, Knight, Bishop, Queen, King
    print(f"inline const int PIECE_VALUES[7] = {{0, {V1[0]}, {V1[1]}, {V1[2]}, {V1[3]}, {V1[4]}, 2000000}};")
    print(f"inline const int u2[5] = {{{V2[0]}, {V2[1]}, {V2[2]}, {V2[3]}, {V2[4]}}};")
    
    print("inline const int u3[7][30] = {")
    print("    {0}, // 空格")
    for i in range(6):
        arr_str = ", ".join(map(lambda x: f"{x:4d}", V3[i]))
        print(f"    {{{arr_str}}}" + ("," if i < 5 else ""))
    print("};")
    
    # V4 吃子威脅：最後的 500000 絕對不能動，這是避免國王自殺的究極防線
    print(f"inline const int u4_tactical[6] = {{{V4[0]}, {V4[1]}, {V4[2]}, {V4[3]}, {V4[4]}, 500000}};")
    print(f"inline const int u5_pawn_stages[4] = {{{V5[0]}, {V5[1]}, {V5[2]}, {V5[3]}}};")
    print(f"constexpr int u6_mobility = {k1};")
    
    v6_str = ", ".join(map(str, V6))
    print(f"inline const int u7_king_step[30] = {{{v6_str}}};")
    print("constexpr int u_intercept = 0;")
    print("// ===============================================================")

if __name__ == '__main__':
    train_model()