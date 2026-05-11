import os
from src.core.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = "processed_data/"
INPUT_FILE = "benchmark_dataset_gemini.csv"
OUTPUT_FILE = "benchmark_dataset_clean.csv"

def main():
    logger.info("Starting CSV cleanup...")

    # 1. Scan data directory to get the list of all valid Champions
    try:
        valid_champs = [f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    except FileNotFoundError:
        logger.error(f"Directory not found: {DATA_DIR}")
        return

    logger.info(f"Loaded {len(valid_champs)} champions for standard filtering.")

    valid_lines = []
    header_found = False
    garbage_count = 0

    # 2. Read each line of the problematic CSV file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Header processing: Keep only 1 header line
        if line.lower().startswith("champion_name"):
            if not header_found:
                valid_lines.append(line)
                header_found = True
            continue

        # 3. Check if this line starts with a valid champion name
        is_valid = False
        for champ in valid_champs:
            # LLM might output: Lee Sin,"Question..." or "Lee Sin","Question..."
            if line.startswith(f'{champ},') or line.startswith(f'"{champ}",'):
                is_valid = True
                break

        if is_valid:
            valid_lines.append(line)
        else:
            garbage_count += 1
            # logger.debug(f"Removed: {line[:50]}...") 

    # 4. Write clean data to a new file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for clean_line in valid_lines:
            f.write(clean_line + "\n")

    logger.info("===========================================")
    logger.info("Cleanup complete!")
    logger.info(f"Retained: {len(valid_lines) - 1} standard data lines (excluding header).")
    logger.info(f"Filtered out: {garbage_count} garbage lines (LLM chatter).")
    logger.info(f"Clean file saved at: {OUTPUT_FILE}")
    logger.info("===========================================")

if __name__ == "__main__":
    main()