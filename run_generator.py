import os
import subprocess
import time
import json

DATA_DIR = "processed_data/"
OUTPUT_FILE = "benchmark_dataset_gemini.csv"
SKILL_FILE = ".gemini/skills/question-generator/SKILL.md"

def load_agent_prompt():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        return f.read()

def main():
    print("🚀 Khởi động luồng tạo Dataset bằng Gemini CLI...")
    
    agent_context = load_agent_prompt()
    
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("champion_name,query,expected_intent,expected_context,ground_truth_answer\n")

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    total_champions = len(files)
    
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
        
        # Đọc dữ liệu tướng để đẩy thẳng vào prompt (tránh việc ép CLI dùng tool đọc file)
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            champ_data = json.load(f)
            
        # Lấy một phần data cần thiết (Ví dụ: Infobox và 1000 ký tự đầu của lore)
        lore_snippet = str(champ_data.get("mainContent", ""))[:1000]
        infobox_snippet = str(champ_data.get("infobox", {}))
        
        # Lắp ghép Prompt hoàn chỉnh
        full_prompt = f"""
{agent_context}

TARGET CHAMPION: {champ_id}

INFOBOX:
{infobox_snippet}

LORE SNIPPET:
{lore_snippet}

Begin generating the 12 CSV lines strictly:
"""

        # Sử dụng cờ -p cho headless mode
        cmd = [
                "gemini",
                "--model", "gemini-2.5-flash-lite", 
                "-p", full_prompt
            ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            output_csv = result.stdout.strip()
            
            # Tách thành các dòng
            lines = output_csv.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                
                # 1. Bỏ qua các dòng trống
                if not line:
                    continue
                    
                # 2. Bỏ qua dòng header nếu LLM tự sinh ra
                if line.lower().startswith('champion_name'):
                    continue
                    
                # 3. LỌC RÁC: Chỉ chấp nhận những dòng BẮT ĐẦU bằng tên tướng hiện tại
                # (Ví dụ: "Lee Sin," hoặc Lee Sin,...)
                if line.startswith(f'"{champ_id}",') or line.startswith(f'{champ_id},'):
                    clean_lines.append(line)
            
            output_csv = "\n".join(clean_lines)
                
            # Trực tiếp ghi vào file bằng Python
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                if output_csv.strip(): 
                    f.write(output_csv + "\n")

            # if output_csv.startswith("```"):
            #     output_csv = "\n".join(output_csv.split("\n")[1:-1])
                
            # lines = output_csv.strip().split('\n')
            # clean_lines = [line for line in lines if not line.lower().startswith('champion_name')]
            # output_csv = "\n".join(clean_lines)
                
            # with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            #     if output_csv.strip(): 
            #         f.write(output_csv + "\n")
                
            print(f"✅ Xong {champ_id}. Đã ghi vào CSV.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy Gemini CLI cho {champ_id}:\n{e.stderr}")
            
        time.sleep(2)

    print("\n🎉 Hoàn tất quá trình tạo Dataset bằng Gemini CLI!")

if __name__ == "__main__":
    main()