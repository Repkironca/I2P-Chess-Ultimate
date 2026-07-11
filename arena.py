import os
import sys
import csv
import json
import time
import random
import subprocess
import itertools
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALGO_TABLE_PATH = os.path.join("build", "algo_table.csv")
LOG_DIR = "logs"
RECORD_DIR = "records"
TASK_FILE = "arena_tasks.json"
MAX_CONSECUTIVE_DRAWS = 3 

for d in [LOG_DIR, RECORD_DIR]:
    os.makedirs(d, exist_ok=True)

class DualLogger:
    def __init__(self, filepath, is_background):
        self.file = open(filepath, 'a', encoding='utf-8')
        self.is_background = is_background
        self.terminal = sys.__stdout__

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        if not self.is_background:
            self.terminal.write(data)
            self.terminal.flush()

class GameCounter:
    def __init__(self):
        self.count = 1

def load_models():
    models = {}
    if not os.path.exists(ALGO_TABLE_PATH): return models
    with open(ALGO_TABLE_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip().lower() != "name" and row[0].strip() != "":
                models[row[0].strip()] = row[1].strip()
    return models

def get_engine_path(name):
    base_path = os.path.join("build", name)
    if sys.platform == "win32":
        return base_path + ".exe" if os.path.exists(base_path + ".exe") else base_path
    return base_path

# ==========================================
# 極簡封裝：直接呼叫改版後的 cli.py
# ==========================================
def play_game(white, black, models, settings, counter, logger):
    idx = counter.count
    counter.count += 1
    
    logger.write(f"\n{'='*60}\n")
    logger.write(f"⚔️ 第 {idx} 局: [白] {white} VS [黑] {black} ... (激戰中)\n")
    logger.write(f"{'='*60}\n")
    
    w_path = get_engine_path(white)
    b_path = get_engine_path(black)
    if not os.path.exists(w_path) or not os.path.exists(b_path):
        logger.write(f"⚠️ 失敗! 找不到執行檔。\n")
        return "skip"

    w_algo = models.get(white, "minimax")
    b_algo = models.get(black, "minimax")

    # [修復] 加上 --verbose 抓取 Moves，加上 --no-board 拒絕垃圾符號
    cmd = [
        sys.executable, "-u", "-m", "cli.cli",
        "--white", w_path,
        "--black", b_path,
        "--time", str(settings['time']),
        "--games", "1",
        "--verbose",
        "--no-board",
        "--white-algo", w_algo,
        "--black-algo", b_algo
    ]

    moves = []
    winner = None

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )

        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            
            if "White:" in line or "Black:" in line or "Result:" in line or "===" in line or "Score after" in line or "wins!" in line or "Draw" in line:
                logger.write(f"  {line}\n")

            if ("White:" in line or "Black:" in line) and ("." in line):
                try:
                    move_part = line.split(":", 1)[1].split("(")[0].strip()
                    move_str = re.sub(r'[^a-zA-Z0-9]', '', move_part).lower()
                    
                    depth = -1
                    depth_match = re.search(r'depth=(\d+)', line)
                    if depth_match: depth = int(depth_match.group(1))
                        
                    if len(move_str) >= 4:
                        moves.append({"move": move_str, "depth": depth})
                except Exception:
                    pass
                    
            elif line.startswith("Result:"):
                res_str = line.split(":", 1)[1].strip()
                if res_str == "1-0": winner = "white"
                elif res_str == "0-1": winner = "black"
                elif res_str == "1/2-1/2": winner = "draw"
            # [修復] 防呆備用：若 Result 抓不到，直接從終局宣言抓勝負
            elif ">> White wins!" in line or ">> Sente wins!" in line:
                winner = "white"
            elif ">> Black wins!" in line or ">> Gote wins!" in line:
                winner = "black"
            elif ">> Draw!" in line:
                winner = "draw"

        proc.wait()

    except Exception as e:
        logger.write(f"⚠️ 失敗! (執行 cli.cli 時發生例外: {e})\n")
        return "skip"

    if winner is None:
        logger.write("⚠️ 警告: 未能解析正常勝負。\n")
        return "skip"

    record = {
        "timestamp": datetime.now().isoformat(),
        "white": white, "black": black,
        "white_algo": w_algo, "black_algo": b_algo,
        "winner": winner, "total_moves": len(moves),
        "moves": moves
    }
    file_name = f"{int(time.time()*1000)}_{white}_vs_{black}.json"
    with open(os.path.join(RECORD_DIR, file_name), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        
    logger.write(f"\n🎉 局數 {idx} 完成! 勝方: {winner} (共 {len(moves)} 步)\n")
    return winner

def run_round_robin(matches, models, settings, counter, logger):
    for w, b in matches: play_game(w, b, models, settings, counter, logger)

def play_match_series(p1, p2, bo_x, rand_first, models, settings, counter, logger):
    wins = {p1: 0, p2: 0}
    target = (bo_x // 2) + 1
    draw_streak = 0
    p1_is_white = random.choice([True, False]) if rand_first else True
    
    while wins[p1] < target and wins[p2] < target:
        w, b = (p1, p2) if p1_is_white else (p2, p1)
        res = play_game(w, b, models, settings, counter, logger)
        if res == "skip": return "skip"
            
        if res == "white":
            wins[p1 if p1_is_white else p2] += 1
            draw_streak = 0
            p1_is_white = not p1_is_white
        elif res == "black":
            wins[p2 if p1_is_white else p1] += 1
            draw_streak = 0
            p1_is_white = not p1_is_white
        else:
            draw_streak += 1
            p1_is_white = not p1_is_white 
            if draw_streak >= MAX_CONSECUTIVE_DRAWS: return "both"
    return p1 if wins[p1] >= target else p2

def run_elimination(participants, models, settings, elim_settings, counter, logger):
    is_double = elim_settings['double_elim']
    bo_x = elim_settings['bo_x']
    rand_first = elim_settings['rand_first']
    losses = {p: 0 for p in participants}
    round_num = 1
    
    while True:
        active = [p for p, l in losses.items() if l < (2 if is_double else 1)]
        if len(active) <= 1: break
            
        logger.write(f"\n--- 🏆 淘汰賽 第 {round_num} 輪 ---\n")
        random.shuffle(active)
        active.sort(key=lambda x: losses[x]) 
        matches = [(active[i], active[i+1]) for i in range(0, len(active)-1, 2)]
            
        for p1, p2 in matches:
            winner = play_match_series(p1, p2, bo_x, rand_first, models, settings, counter, logger)
            if winner == "both": pass
            elif winner == p1: losses[p2] += 1
            elif winner == p2: losses[p1] += 1
            else: losses[p1] += 1; losses[p2] += 1
        round_num += 1
        
    champion = [p for p, l in losses.items() if l < (2 if is_double else 1)]
    if champion: logger.write(f"\n🎉 淘汰賽最終冠軍：{', '.join(champion)}！\n\n")

def run_custom(matches, models, settings, counter, logger):
    for match in matches:
        w, b, k, swap = match['w'], match['b'], match['k'], match['swap']
        for _ in range(k): play_game(w, b, models, settings, counter, logger)
        if swap:
            for _ in range(k): play_game(b, w, models, settings, counter, logger)

def run_worker(is_background=False):
    if not os.path.exists(TASK_FILE): return
    with open(TASK_FILE, 'r', encoding='utf-8') as f: task_data = json.load(f)
    log_file = os.path.join(LOG_DIR, f"arena_run_{int(time.time())}.log")
    logger = DualLogger(log_file, is_background)
    
    logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Arena 任務開始！(日誌: {log_file})\n")
    models, mode, settings = task_data['models'], task_data['mode'], task_data['settings']
    counter = GameCounter()
    
    if mode == 'round_robin': run_round_robin(task_data['matches'], models, settings, counter, logger)
    elif mode == 'elimination': run_elimination(task_data['participants'], models, settings, task_data['elim_settings'], counter, logger)
    elif mode == 'custom': run_custom(task_data['matches'], models, settings, counter, logger)

    logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 所有對戰任務已執行完畢！\n")
    try: os.remove(TASK_FILE) 
    except: pass

def get_input(prompt, valid_options=None, cast_type=str, min_val=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val: continue
            val = cast_type(val)
            if valid_options and val not in valid_options: continue
            if min_val is not None and val < min_val: continue
            return val
        except ValueError: pass

def get_yes_no(prompt):
    return get_input(prompt + " (y/n): ", ['y', 'n', 'Y', 'N']).lower() == 'y'

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        run_worker(is_background=True)
        return

    models = load_models()
    if not models: return
        
    while True:
        print("\n" + "="*50)
        print("🛡️  Mini Chess 終極大亂鬥 Arena  🛡️")
        print("="*50)
        print("  [1] ⚔️ 車輪戰 (Round Robin)")
        print("  [2] 🏆 淘汰賽 (Elimination)")
        print("  [3] 🎯 指定賽 (Custom Match)")
        print("  [0] 🚪 離開")
        
        choice = get_input("➤ 輸入選項 (0-3): ", ['0', '1', '2', '3'])
        if choice == '0': break
            
        task_data = {"models": models, "settings": {"time": 2000}}
        
        if choice == '1':
            k = get_input("  每組對手互相對戰次數 (K)? ", cast_type=int, min_val=1)
            swap = get_yes_no("  是否交換黑白重賽?")
            matches = []
            for w, b in itertools.combinations(list(models.keys()), 2):
                for _ in range(k): matches.append((w, b))
                if swap:
                    for _ in range(k): matches.append((b, w))
            task_data.update({"mode": "round_robin", "matches": matches})
            
        elif choice == '2':
            is_double = get_yes_no("  是否採用雙敗淘汰制?")
            bo_x = get_input("  每場採幾戰幾勝 (BoX)? ", cast_type=int, min_val=1)
            rand_first = get_yes_no("  隨機先後手?")
            task_data.update({
                "mode": "elimination", "participants": list(models.keys()),
                "elim_settings": {"double_elim": is_double, "bo_x": bo_x, "rand_first": rand_first}
            })
            
        elif choice == '3':
            custom_matches = []
            while True:
                cmd = input("  ➤ 輸入 (白方 黑方 局數 是否交換[y/n]) 或 'done': ").strip()
                if cmd.lower() == 'done': break
                if cmd.lower() == 'list':
                    print("  名單:", ", ".join(models.keys()))
                    continue
                parts = cmd.split()
                if len(parts) == 4 and parts[0] in models and parts[1] in models:
                    custom_matches.append({"w": parts[0], "b": parts[1], "k": int(parts[2]), "swap": parts[3].lower() in ['y', 'yes']})
            if not custom_matches: continue
            task_data.update({"mode": "custom", "matches": custom_matches})
            
        is_unstop = get_yes_no("\n🚀 是否開啟 Unstop 模式?")
        with open(TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
            
        if is_unstop:
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} if sys.platform == "win32" else {"start_new_session": True}
            worker_proc = subprocess.Popen([sys.executable, __file__, '--worker'], **kwargs)
            print(f"✅ 任務已送入背景！(PID: {worker_proc.pid})")
            break
        else:
            run_worker(is_background=False)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()