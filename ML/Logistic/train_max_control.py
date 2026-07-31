import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score

def train_model():
    data = np.load('features.npz')
    X = data['X']
    y = data['y']
    steps = data['steps']
    
    # 權重設定：和局 0.2，勝負 1.0，並加上 step^0.3
    base_w = np.where(y == 0, 0.2, 1.0)
    w = base_w * (steps ** 0.3)
    
    # LSQ Linear 需要將 weight 乘入矩陣中
    W_sqrt = np.sqrt(w)
    A = X * W_sqrt[:, np.newaxis]
    b = y * W_sqrt
    
    # 設定變數的上下界 (共 230 個變數)
    # V1 (5): P, N, B, R, Q
    lb_V1 = [100, 250, 250, 450, 800]
    ub_V1 = [200, 450, 450, 750, 1200]
    
    # V2 (5): Manhattan
    lb_V2 = [-20] * 5
    ub_V2 = [20] * 5
    
    # V3 (180): PST
    lb_V3 = [-50] * 180
    ub_V3 = [50] * 180
    
    # V5 (4): Pawn Stages
    lb_V5 = [-40] * 4
    ub_V5 = [40] * 4
    
    # V6 (30): King Step
    lb_V6 = [-5] * 30
    ub_V6 = [5] * 30
    
    # V4 (5): Tactical Threats (P~Q)
    lb_V4 = [0] * 5
    ub_V4 = [60] * 5
    
    # k1 (1): Mobility
    lb_k1 = [0]
    ub_k1 = [15]
    
    lb = np.concatenate([lb_V1, lb_V2, lb_V3, lb_V5, lb_V6, lb_V4, lb_k1])
    ub = np.concatenate([ub_V1, ub_V2, ub_V3, ub_V5, ub_V6, ub_V4, ub_k1])
    
    print("Optimization started. This might take a moment...")
    res = lsq_linear(A, b, bounds=(lb, ub), max_iter=200)
    
    coef = np.round(res.x).astype(int)
    
    # 計算 R^2
    y_pred = X @ res.x
    r2 = r2_score(y, y_pred, sample_weight=w)
    print(f"// R^2 Score (Weighted): {r2:.3f}")
    
    # 提取並格式化輸出為 C++ 程式碼
    V1 = coef[0:5]
    V2 = coef[5:10]
    V3 = coef[10:190].reshape(6, 30)
    V5 = coef[190:194]
    V6 = coef[194:224]
    V4 = coef[224:229]
    k1 = coef[229]
    
    print("\n// ================== C++ 參數區 (直接複製貼上) ==================")
    print(f"inline const int PIECE_VALUES[7] = {{0, {V1[0]}, {V1[1]}, {V1[2]}, {V1[3]}, {V1[4]}, 2000000}};")
    print(f"inline const int u2[5] = {{{V2[0]}, {V2[1]}, {V2[2]}, {V2[3]}, {V2[4]}}};")
    
    print("inline const int u3[7][30] = {")
    print("    {0}, // 空格")
    for i in range(6):
        arr_str = ", ".join(map(str, V3[i]))
        print(f"    {{{arr_str}}}" + ("," if i < 5 else ""))
    print("};")
    
    print(f"inline const int u4_tactical[6] = {{{V4[0]}, {V4[1]}, {V4[2]}, {V4[3]}, {V4[4]}, 500000}};")
    print(f"inline const int u5_pawn_stages[4] = {{{V5[0]}, {V5[1]}, {V5[2]}, {V5[3]}}};")
    print(f"constexpr int u6_mobility = {k1};")
    
    v6_str = ", ".join(map(str, V6))
    print(f"inline const int u7_king_step[30] = {{{v6_str}}};")
    print("constexpr int u_intercept = 0; // lsq_linear 無截距項，設為0")
    print("// ===============================================================")

if __name__ == '__main__':
    train_model()