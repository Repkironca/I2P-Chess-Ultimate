import os
import glob
import subprocess
import re

def extract_algorithms(exe_path):
    try:
        # 使用 subprocess.run 執行並一次性給予 "ubgi\nquit\n" 指令
        # 加上 timeout 避免同學的程式寫壞導致無限卡死
        proc = subprocess.run(
            [exe_path],
            input="ubgi\nquit\n",
            capture_output=True,
            text=True,
            timeout=2,  # 2秒沒反應就直接放棄該程式
            encoding='utf-8',
            errors='ignore' # 忽略編碼錯誤，以防有人輸出了奇怪的字元
        )
        
        default_algo = None
        algos = []
        
        # 解析引擎吐出來的資訊
        for line in proc.stdout.splitlines():
            line = line.strip()
            
            # 尋找 Algorithm 這行選項
            if line.startswith("option name Algorithm"):
                # 抓取 default 後面的名字
                default_match = re.search(r'default\s+(\S+)', line)
                if default_match:
                    default_algo = default_match.group(1)
                    
                # 抓取所有 var 後面的名字
                algos = re.findall(r'var\s+(\S+)', line)
                break # 找到就不用繼續往下看了
                
        return default_algo, algos
        
    except subprocess.TimeoutExpired:
        return None, ["Error: 執行逾時 (Timeout)"]
    except Exception as e:
        return None, [f"Error: {str(e)}"]

def main():
    # 抓取當前目錄下所有的 .exe 檔案
    exe_files = glob.glob("*.exe")
    
    if not exe_files:
        print("❌ 找不到任何 .exe 檔案，請確認腳本是否與同學的執行檔放在同個資料夾。")
        return

    output_filename = "algo_table.txt"
    
    print(f"⏳ 開始掃描 {len(exe_files)} 個執行檔，請稍候...\n")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        # 寫入標頭
        f.write(f"{'執行檔名稱':<25} | {'預設演算法':<20} | {'所有可用演算法'}\n")
        f.write("-" * 80 + "\n")
        
        for exe in exe_files:
            exe_name = os.path.basename(exe)
            print(f"🔍 正在解析: {exe_name}")
            
            default_algo, algos = extract_algorithms(exe)
            
            if default_algo is None and (not algos or "Error" in algos[0]):
                error_msg = algos[0] if algos else "Error: 無法取得 Algorithm 選項"
                f.write(f"{exe_name:<25} | {'[解析失敗]':<20} | {error_msg}\n")
            else:
                default_str = default_algo if default_algo else "N/A"
                algos_str = ", ".join(algos) if algos else "N/A"
                f.write(f"{exe_name:<25} | {default_str:<20} | {algos_str}\n")
                
    print(f"\n✅ 掃描完成！結果已成功儲存至 {output_filename}")

if __name__ == "__main__":
    main()