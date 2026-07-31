import os
import glob
import json
import csv

# 棋子與棋盤大小
BOARD_W = 5
BOARD_H = 6

# 輸入與輸出路徑
input_dir = "../records"          # 存放 JSON 紀錄的目錄
output_dir = "extracted"       # CSV 檔案輸出的目錄

def init_board():
    """初始化 MiniChess 起始盤面"""
    return [
        ['k', 'q', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K']
    ]

def parse_sq(sq_str):
    """將 UCI 座標 (如 c2) 轉換為 2D 陣列索引 (row, col)"""
    col = ord(sq_str[0]) - ord('a')
    # 假設白方在下方 (row 5)，黑方在上方 (row 0)
    # a1 對應到 row 5, col 0
    row = BOARD_H - int(sq_str[1])
    return row, col

def board_to_fen(board):
    """將 2D 陣列轉換為簡化的 FEN 字串格式"""
    rows = []
    for r in board:
        empty_count = 0
        row_str = ""
        for cell in r:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += cell
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)
    return "/".join(rows)

def apply_move(board, move_str):
    """在虛擬棋盤上執行移動，包含兵的升變邏輯"""
    if len(move_str) < 4:
        return
        
    fr, fc = parse_sq(move_str[0:2])
    tr, tc = parse_sq(move_str[2:4])
    
    piece = board[fr][fc]
    board[fr][fc] = '.'

    # 升變邏輯 (Promotion)
    # 白兵 ('P') 走到第 0 列變成后 ('Q')
    # 黑兵 ('p') 走到第 5 列變成后 ('q')
    if piece == 'P' and tr == 0:
        piece = 'Q'
    elif piece == 'p' and tr == BOARD_H - 1:
        piece = 'q'

    board[tr][tc] = piece

def main():
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset.csv")

    if not os.path.exists(input_dir):
        print(f"⚠️ 找不到目錄 '{input_dir}'，請確認是否有產生對戰紀錄。")
        return

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        print(f"⚠️ 在 '{input_dir}' 中找不到任何 JSON 檔案。")
        return

    print(f"🔍 找到 {len(json_files)} 份對戰紀錄，開始正規化特徵...")

    # 開啟 CSV 檔案準備寫入
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 寫入 Header，加入你要求保留的 Meta 資訊
        writer.writerow([
            "game_id", 
            "timestamp", 
            "white_model", 
            "black_model", 
            "step", 
            "player", 
            "board_fen", 
            "result"
        ])

        processed_games = 0
        total_positions = 0

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                # 提取你指定的 Meta 資訊
                game_id = os.path.basename(file_path)
                timestamp = data.get("timestamp", "")
                white_model = data.get("white", "unknown")
                black_model = data.get("black", "unknown")
                winner = data.get("winner", "draw")

                # 結果對白方而言：贏=1.0, 輸=-1.0, 和局=0.0
                if winner == "white":
                    res_val = 1.0
                elif winner == "black":
                    res_val = -1.0
                else:
                    res_val = 0.0

                board = init_board()
                current_player = 'w' # 白方先下
                step = 1

                # 解析每一步棋
                for move_info in data.get("moves", []):
                    move_str = move_info.get("move", "")
                    if not move_str:
                        continue
                    
                    # 紀錄當前「移動前」的盤面狀態
                    fen = board_to_fen(board)
                    writer.writerow([
                        game_id,
                        timestamp,
                        white_model,
                        black_model,
                        step,
                        current_player,
                        fen,
                        res_val
                    ])
                    total_positions += 1

                    # 執行該步棋，更新虛擬棋盤
                    apply_move(board, move_str)

                    # 切換玩家與遞增步數
                    current_player = 'b' if current_player == 'w' else 'w'
                    step += 1
                
                processed_games += 1

            except Exception as e:
                print(f"⚠️ 處理檔案 {file_path} 時發生錯誤: {e}")

    print("-" * 40)
    print("✅ 資料正規化完成！")
    print(f"總共處理對局數 : {processed_games}")
    print(f"總共產出盤面數 : {total_positions}")
    print(f"資料已儲存至   : {output_file}")

if __name__ == "__main__":
    main()