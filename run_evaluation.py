import os
import json
import time
import subprocess
import pandas as pd
from tqdm import tqdm
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Path configuration
INPUT_JSONL = "rag_outputs.jsonl"
OUTPUT_CSV = "evaluation_results_final.csv"
SKILL_FILE = ".gemini/skills/rag-evaluator/SKILL.md"

# ==========================================
# 1. HELPERS
# ==========================================
def load_evaluator_skill():
    if not os.path.exists(SKILL_FILE):
        logger.error(f"Error: Agent Skill file not found at {SKILL_FILE}")
        return ""
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content.strip()

def calculate_retrieval_metrics(retrieved_chunks, expected_context, k=3):
    """Calculates Precision@K, Recall@K, F1@K"""
    expected_raw = str(expected_context).lower()

    # Adversarial cases (Trap questions, no info)
    if expected_raw in ["n/a", "nan", ""] or "no explicit mention" in expected_raw:
        # If RAG also returns no chunks (or empty chunks) -> Perfect!
        if not retrieved_chunks or len(str(retrieved_chunks[0]).strip()) < 10:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0 # RAG "hallucinated" and retrieved garbage info

    # Get top K chunks
    top_k_chunks = retrieved_chunks[:k]
    expected_keywords = [kw.strip() for kw in expected_raw.split(',')]

    relevant_count = 0
    for chunk in top_k_chunks:
        chunk_lower = str(chunk).lower()
        # If chunk contains any keyword -> Count as Relevant
        if any(kw in chunk_lower for kw in expected_keywords):
            relevant_count += 1

    total_expected = len(expected_keywords)

    precision = relevant_count / k if k > 0 else 0.0
    recall = relevant_count / total_expected if total_expected > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_with_gemini(skill_context, ground_truth, actual_answer):
    """Call Gemini CLI to score the answer from 0-10"""
    full_prompt = f"""
{skill_context}

Activate 'rag-evaluator' skill.

--- GROUND TRUTH ---
{ground_truth}

--- ACTUAL ANSWER ---
{actual_answer}

Score (0-10):
"""
    cmd = ["gemini", "--model", "gemini-2.5-flash-lite"]

    try:
        result = subprocess.run(
            cmd, 
            input=full_prompt,
            capture_output=True, 
            text=True, 
            check=True, 
            encoding="utf-8"
        )

        raw_output = result.stdout.strip()
        clean_score = ''.join(filter(str.isdigit, raw_output))
        if clean_score:
            return min(max(int(clean_score), 0), 10)
        return 0
    except Exception as e:
        logger.warning(f"Error calling Gemini CLI Judge: {e}")
        return None

# ==========================================
# 2. MAIN EVALUATION FLOW
# ==========================================
def main():
    logger.info("Starting Evaluation flow (Independent Scoring)...")

    if not os.path.exists(INPUT_JSONL):
        logger.error(f"Inference data file not found: {INPUT_JSONL}")
        logger.info("💡 Please run run_rag_inference.py first!")
        return

    skill_context = load_evaluator_skill()
    if not skill_context: return

    # --- LOGIC CHECKPOINT ---
    processed_queries = set()
    file_exists = os.path.exists(OUTPUT_CSV)

    if file_exists:
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if 'query' in existing_df.columns:
                processed_queries = set(existing_df['query'].tolist())
                logger.info(f"⏩ Found {len(processed_queries)} already scored queries. Resuming remaining ones...")
        except Exception:
            pass

    # Read all data from JSONL
    inference_records = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                inference_records.append(json.loads(line))

    logger.info(f"🔎 Starting scoring for {len(inference_records)} records...")

    for record in tqdm(inference_records):
        query = record['query']

        if query in processed_queries:
            continue

        # 1. Measure Routing
        expected_intent = str(record['expected_intent']).lower()
        actual_intent = str(record['actual_intent']).lower()
        routing_score = 1 if expected_intent == actual_intent else 0

        # 2. Measure Retrieval (Precision@3, Recall@3, F1@3)
        retrieved_chunks = record.get('retrieved_chunks', [])
        expected_context = record.get('expected_context', '')
        p3, r3, f1_3 = calculate_retrieval_metrics(retrieved_chunks, expected_context, k=3)

        # 3. Measure Generation (LLM-as-a-Judge)
        gen_score = evaluate_with_gemini(
            skill_context, 
            record['ground_truth_answer'], 
            record['actual_answer']
        )

        # Handle Quota exhausted
        if gen_score is None:
            logger.error("🛑 System error (possibly API Quota exhausted). Script will pause to preserve data.")
            break

        # Package results
        row_result = {
            "query": query,
            "expected_intent": record['expected_intent'],
            "actual_intent": record['actual_intent'],
            "routing_score": routing_score,
            "precision@3": round(p3, 2),
            "recall@3": round(r3, 2),
            "f1@3": round(f1_3, 2),
            "gen_score": gen_score,
            "latency": record['latency']
        }

        # Save checkpoint immediately
        res_df = pd.DataFrame([row_result])
        res_df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
        file_exists = True 

        time.sleep(1) # Prevent Rate Limit

    # ==========================================
    # 3. FINAL REPORT SUMMARY
    # ==========================================
    if os.path.exists(OUTPUT_CSV):
        final_df = pd.read_csv(OUTPUT_CSV)
        report = {
            "Routing Accuracy (%)": final_df['routing_score'].mean() * 100,
            "Avg Precision@3": final_df['precision@3'].mean(),
            "Avg Recall@3": final_df['recall@3'].mean(),
            "Avg F1@3": final_df['f1@3'].mean(),
            "Avg Gen Score (0-10)": final_df['gen_score'].mean(),
            "Avg Latency (s)": final_df['latency'].mean()
        }

        print("\n" + "="*45)
        print("🏆 RAG SYSTEM PERFORMANCE REPORT (FINAL)")
        print("="*45)
        for metric, value in report.items():
            print(f"{metric:30}: {value:.2f}")
        print("="*45)

if __name__ == "__main__":
    main()