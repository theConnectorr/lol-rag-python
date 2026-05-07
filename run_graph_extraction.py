import os
import json
import subprocess
import time

DATA_DIR = "processed_data/"
SKILL_FILE = ".gemini/skills/graph-extractor/SKILL.md"

def load_skill_prompt():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        # Bỏ qua phần YAML frontmatter (từ --- đến ---) khi đưa vào prompt 
        # để LLM tập trung 100% vào Instructions
        content = f.read()
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content.strip()

def extract_json_from_text(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"): lines = lines[1:]
        if lines[-1].startswith("```"): lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

def main():
    print("🚀 Khởi động luồng trích xuất Knowledge Graph với chuẩn Agent Skills...")
    skill_context = load_skill_prompt()
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    total_champions = len(files)
    
    for idx, filename in enumerate(files):
        champ_id = filename.replace('.json', '')
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            champ_data = json.load(f)
            
        if "knowledge_graph" in champ_data and champ_data["knowledge_graph"]:
            print(f"⏩ Bỏ qua {champ_id} (Đã có Knowledge Graph).")
            continue
            
        print(f"\n⏳ [{idx+1}/{total_champions}] Phân tích Graph cho {champ_id}...")
        
        lore_snippet = str(champ_data.get("mainContent", ""))[:2500] 
        
        # Gọi thẳng tên skill trong prompt để nhấn mạnh
        full_prompt = f"""
{skill_context}

Activate 'graph-extractor' skill for the following data.
TARGET CHAMPION: {champ_id}

LORE TEXT:
{lore_snippet}

Begin JSON extraction strictly:
"""

        cmd = [
            "gemini", 
            "--model", "gemini-2.5-flash", 
            "-p", full_prompt
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            raw_output = result.stdout
            
            clean_json_str = extract_json_from_text(raw_output)
            graph_data = json.loads(clean_json_str)
            
            champ_data["knowledge_graph"] = graph_data
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(champ_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Xong {champ_id}. Đã trích xuất được {len(graph_data.get('nodes', []))} nodes và {len(graph_data.get('edges', []))} edges.")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi CLI khi chạy {champ_id}:\n{e.stderr}")
        except json.JSONDecodeError as e:
            print(f"❌ Lỗi parse JSON từ kết quả của {champ_id}:\nOutput thô: {clean_json_str}")
            
        time.sleep(1)

    print("\n🎉 Hoàn tất quá trình xây dựng Graph!")

if __name__ == "__main__":
    main()