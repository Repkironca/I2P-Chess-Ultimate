import os
import json
import pandas as pd
import glob
import shutil

# ==========================================
# 嚴格對齊順序：空格(0), 小兵(1), 騎士(2), 主教(3), 城堡(4), 皇后(5), 國王(6)
# 白方為正數，黑方為負數
# 對齊 MinitChess 初始盤面
# ==========================================
INITIAL_BOARD = [
    [-4, -2, -3, -5, -6], # Row 0 (Rank 6) 黑方: r(a6), n(b6), b(c6), q(d6), k(e6)
    [-1, -1, -1, -1, -1], # Row 1 (Rank 5) 黑方 Pawn
    [ 0,  0,  0,  0,  0], # Row 2 (Rank 4)
    [ 0,  0,  0,  0,  0], # Row 3 (Rank 3)
    [ 1,  1,  1,  1,  1], # Row 4 (Rank 2) 白方 Pawn
    [ 4,  2,  3,  5,  6]  # Row 5 (Rank 1) 白方: R(a1), N(b1), B(c1), Q(d1), K(e1)
]

def create_and_clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def uci_to_coords(uci):
    # 將 c2c3 這類字串轉為二維陣列的 row, col
    from_c = ord(uci[0]) - ord('a')
    from_r = 6 - int(uci[1])
    to_c = ord(uci[2]) - ord('a')
    to_r = 6 - int(uci[3])
    return from_r, from_c, to_r, to_c

def extract_games(records_dir="../records", output_dir="../ML/extracted"):
    create_and_clear_dir(output_dir)
    output_csv = os.path.join(output_dir, "dataset.csv")
    
    all_features = []
    json_files = glob.glob(os.path.join(records_dir, "*.json"))
    
    print(f"Found {len(json_files)} game records. Extracting...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                
            # 讀取勝負
            winner_str = game_data.get("winner", "draw")
            if winner_str == "white":
                game_result = 1.0
            elif winner_str == "black":
                game_result = 0.0
            else:
                game_result = 0.5
                
            white_player = game_data.get("white", "unknown")
            black_player = game_data.get("black", "unknown")
            timestamp = game_data.get("timestamp", "")
            moves = game_data.get("moves", [])
            
            # 初始化 5x6 棋盤副本
            board = [row[:] for row in INITIAL_BOARD]
            
            # 依序執行每一手
            for step, move_info in enumerate(moves):
                # 攤平盤面為一維陣列 (長度 30)
                flat_board = [piece for row in board for piece in row]
                current_player = 0 if step % 2 == 0 else 1
                
                row_data = {
                    "timestamp": timestamp,
                    "game_id": os.path.basename(file_path),
                    "white_player": white_player,
                    "black_player": black_player,
                    "step": step,
                    "current_player": current_player,
                    "result": game_result
                }
                
                # 寫入 30 個格子的狀態
                for i in range(30):
                    row_data[f"sq_{i}"] = flat_board[i]
                    
                all_features.append(row_data)
                
                # 更新棋盤以供下一回合萃取
                uci = move_info.get("move", "")
                if not uci or uci == "0000":
                    break
                    
                fr, fc, tr, tc = uci_to_coords(uci)
                
                piece = board[fr][fc]
                board[fr][fc] = 0
                board[tr][tc] = piece
                
                # 小兵升變判斷
                if piece == 1 and tr == 0:
                    board[tr][tc] = 5
                elif piece == -1 and tr == 5:
                    board[tr][tc] = -5
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"Extraction complete! Saved {len(df)} samples to {output_csv}.")
    else:
        print("Failed to extract any samples.")

if __name__ == "__main__":
    extract_games()