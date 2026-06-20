import subprocess
import random
import sys
import csv
import os
import time
import re

# ================= 配置區 =================
OUTPUT_FILE = 'dataset_ultra.csv'
TIME_LIMIT = 1800

# 定義要對戰的引擎與其參數
BOTS = [
    {"name": "BOSS", "path": "build/boss-ubgi.exe"},
    {"name": "Lucifer", "path": "build/Lucifer.exe"},
    {"name": "LinearReg", "path": "build/minichess-ubgi.exe"}
]
# ==========================================

def save_result(result, white_name, black_name, moves):
    row = [result, white_name, black_name, " ".join(moves)]
    file_exists = os.path.exists(OUTPUT_FILE)
    
    # 無懼 Excel 鎖定的寫入迴圈
    while True:
        try:
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Result', 'White', 'Black', 'Moves'])
                writer.writerow(row)
            break
        except PermissionError:
            print(f"\n⚠️ 警告: {OUTPUT_FILE} 被鎖定 (可能正用 Excel 開啟)。")
            print("等待 2 秒後將自動重試，請將檔案切換為唯讀或關閉檔案...")
            time.sleep(2)

def play_game(game_idx):
    # 隨機抽取兩支 AI，並決定黑白
    white_bot, black_bot = random.sample(BOTS, 2)

    # 組合指令: python -m cli.cli --white <...> --black <...> --time 1800 --games 1
    cmd = [
        sys.executable, "-m", "cli.cli",
        "--white", white_bot["path"],
        "--black", black_bot["path"],
        "--time", str(TIME_LIMIT),
        "--games", "1"
    ]

    # 利用 --param 傳遞 Algorithm 參數給 minichess-ubgi，避免動到 cli.py 的 argparse choices
    if "minichess-ubgi" in white_bot["path"]:
        cmd.extend(["--white-param", "Algorithm=linear_regression"])
    if "minichess-ubgi" in black_bot["path"]:
        cmd.extend(["--black-param", "Algorithm=linear_regression"])

    print(f"⚔️ 第 {game_idx} 局: [白] {white_bot['name']} VS [黑] {black_bot['name']} ... ", end="", flush=True)

    try:
        # 啟動 CLI 並擷取輸出 (capture_output=True 會隱藏 CLI 的洗畫面)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout

        moves = []
        game_result = None

        # 逐行解析 cli.py 的標準輸出
        for line in output.splitlines():
            line = line.strip()
            
            # 擷取走步紀錄 (例如 "1. White: a2a3 (depth=...)" 或 "1... Black: B2->B3")
            if ("White:" in line or "Black:" in line) and ("." in line):
                try:
                    # 取出冒號後的字串並去掉括號內的 info
                    move_part = line.split(":", 1)[1].split("(")[0].strip()
                    # 把可能有符號的格式 (如 A2->A3) 乾淨地轉回 UCI 小寫 (a2a3)
                    move_str = re.sub(r'[^a-zA-Z0-9]', '', move_part).lower()
                    if len(move_str) >= 4:
                        moves.append(move_str)
                except Exception:
                    pass
            
            # 擷取終局勝負 (例如 "Result: 1-0")
            elif line.startswith("Result:"):
                res_str = line.split(":", 1)[1].strip()
                if res_str == "1-0":
                    game_result = 1.0
                elif res_str == "0-1":
                    game_result = 0.0
                elif res_str == "1/2-1/2":
                    game_result = 0.5

        # 寫入 CSV
        if game_result is not None and len(moves) > 0:
            print(f"完成! 結果: {game_result} (共 {len(moves)} 步)")
            save_result(game_result, white_bot["name"], black_bot["name"], moves)
        else:
            print("失敗! (未能解析結果，可能是引擎提早崩潰)")

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"\n執行發生錯誤: {e}")

if __name__ == '__main__':
    print(f"🚀 開始無盡對戰模式！(使用內建 cli.cli，時限: {TIME_LIMIT}ms / 步)")
    print(f"💾 結果將儲存至: {OUTPUT_FILE} (中途可隨時用唯讀模式開啟檢視)")
    print("🛑 按下 Ctrl+C 即可安全停止\n")
    
    game_counter = 1
    try:
        while True:
            play_game(game_counter)
            game_counter += 1
    except KeyboardInterrupt:
        print("\n\n🛑 收到終止訊號 (Ctrl+C)，安全退出系統。辛苦了！")
        sys.exit(0)