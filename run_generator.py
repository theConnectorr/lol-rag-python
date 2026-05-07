# run_generator.py
import os
import subprocess
import time

DATA_DIR = "processed_data/"
OUTPUT_FILE = "benchmark_dataset_gemini.csv"

def main():
    print("🚀 Khởi động luồng tạo Dataset bằng Gemini CLI...")
    
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("champion,query,expected_intent,expected_context,ground_truth_answer\n")
            
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    total_champions = len(files)
    
    # Đọc các tướng đã xử lý để có thể resume
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            processed_champs = set(line.split(',')[0] for line in lines[1:] if line.strip())
    except Exception:
        processed_champs = set()

    for idx, filename in enumerate(files):
        champ_id = filename.replace('.json', '')
        if champ_id in processed_champs:
            print(f"⏩ Bỏ qua {champ_id} (Đã có trong dataset).")
            continue
            
        print(f"\n⏳ [{idx+1}/{total_champions}] Gọi Gemini CLI xử lý {champ_id}...")
        
        # Lệnh gọi Gemini CLI headlessly
        cmd = [
            "gemini", 
            "-a", "dataset_generator", 
            "-m", f"Generate the dataset for {champ_id}"
        ]
        
        try:
            # Chạy subprocess đồng bộ
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Xong {champ_id}. Output CLI:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy Gemini CLI cho {champ_id}:\n{e.stderr}")
            
        # Nghỉ 2 giây để tránh rate limit của API
        time.sleep(2)

    print("\n🎉 Hoàn tất quá trình tạo Dataset bằng Gemini CLI!")

if __name__ == "__main__":
    main()
