import os
import subprocess
import time
import json
from src.core.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = "processed_data/"
OUTPUT_FILE = "benchmark_dataset_gemini.csv"
SKILL_FILE = ".gemini/skills/question-generator/SKILL.md"

def load_agent_prompt():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        return f.read()

def main():
    logger.info("Starting Dataset generation flow using Gemini CLI...")

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
            logger.info(f"⏩ Skipping {champ_id} (Already in dataset).")
            continue

        logger.info(f"⏳ [{idx+1}/{total_champions}] Calling Gemini CLI to process {champ_id}...")

        # Read champion data to inject into prompt (avoiding CLI tool call for reading file)
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            champ_data = json.load(f)

        # Extract necessary data (e.g., Infobox and first 1000 characters of lore)
        lore_snippet = str(champ_data.get("mainContent", ""))[:1000]
        infobox_snippet = str(champ_data.get("infobox", {}))

        # Construct full Prompt
        full_prompt = f"""
{agent_context}

TARGET CHAMPION: {champ_id}

INFOBOX:
{infobox_snippet}

LORE SNIPPET:
{lore_snippet}

Begin generating the 12 CSV lines strictly:
"""

        # Use -p flag for headless mode
        cmd = [
                "gemini",
                "--model", "gemini-2.5-flash-lite", 
                "-p", full_prompt
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            output_csv = result.stdout.strip()

            # Split into lines
            lines = output_csv.split('\n')
            clean_lines = []

            for line in lines:
                line = line.strip()

                # 1. Skip empty lines
                if not line:
                    continue

                # 2. Skip header line if LLM generated it
                if line.lower().startswith('champion_name'):
                    continue

                # 3. GARBAGE FILTER: Only accept lines starting with the current champion name
                if line.startswith(f'"{champ_id}",') or line.startswith(f'{champ_id},'):
                    clean_lines.append(line)

            output_csv = "\n".join(clean_lines)

            # Write directly to file
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                if output_csv.strip(): 
                    f.write(output_csv + "\n")

            logger.info(f"✅ Finished {champ_id}. Written to CSV.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error running Gemini CLI for {champ_id}:\n{e.stderr}")

        time.sleep(2)

    logger.info("Dataset generation process via Gemini CLI complete!")

if __name__ == "__main__":
    main()