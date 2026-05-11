import os
import json
import subprocess
import time
from src.core.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = "processed_data/"
SKILL_FILE = ".gemini/skills/graph-extractor/SKILL.md"

def load_skill_prompt():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        # Skip YAML frontmatter (between --- and ---) when injecting into prompt
        # to let LLM focus 100% on Instructions
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
    logger.info("Starting Knowledge Graph extraction flow with standard Agent Skills...")
    skill_context = load_skill_prompt()
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    total_champions = len(files)
    
    for idx, filename in enumerate(files):
        champ_id = filename.replace('.json', '')
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            champ_data = json.load(f)
            
        if "knowledge_graph" in champ_data and champ_data["knowledge_graph"]:
            logger.info(f"⏩ Skipping {champ_id} (Knowledge Graph already exists).")
            continue
            
        logger.info(f"⏳ [{idx+1}/{total_champions}] Analyzing Graph for {champ_id}...")
        
        lore_snippet = str(champ_data.get("mainContent", ""))[:2500] 
        
        # Explicitly mention the skill name in prompt for emphasis
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
                
            logger.info(f"✅ Finished {champ_id}. Extracted {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ CLI error while processing {champ_id}:\n{e.stderr}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON from {champ_id} results:\nRaw output: {clean_json_str}")
            
        time.sleep(1)

    logger.info("Graph construction process complete!")

if __name__ == "__main__":
    main()