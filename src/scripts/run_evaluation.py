import os
import json
import time
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Path configuration
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

def evaluate_retrieval_with_gemini(query: str, retrieved_chunks: list, expected_context: str) -> bool:
    """
    Use Gemini CLI as a judge to evaluate whether the 
    retrieved information helps answering the question or not
    (Retrieval Relevance)
    """
    if not retrieved_chunks:
        return False
        
    doc_string = "\n\n".join(str(chunk) for chunk in retrieved_chunks)
    
    prompt = f"""
You are a strict evaluator. You are given a QUESTION, a set of retrieved FACTS and the EXPECTED CONTEXT that the FACTS should mention.
Your goal is to determine if the FACTS contain ANY semantic meaning or information that helps answer the QUESTION.

QUESTION: {query}
FACTS: {doc_string}
EXPECTED CONTEXT: {expected_context}

Return EXACTLY ONE WORD: "True" if the facts are relevant and helpful, or "False" if they are completely unrelated.
"""
    cmd = ["gemini", "--model", "gemini-2.5-flash-lite"]
    
    try:
        result = subprocess.run(
            cmd, 
            input=prompt,
            capture_output=True, 
            text=True, 
            check=True, 
            encoding="utf-8"
        )
        
        answer = result.stdout.strip().lower()
        return "true" in answer
    except Exception as e:
        print(f"⚠️ Lỗi gọi Gemini Judge (Retrieval): {e}")
        return False
    
def evaluate_groundedness_with_gemini(retrieved_chunks: list, actual_answer: str) -> bool:
    """
    Hallucination judge.
    Ensure there's no answer using information beside the provided context.
    """
    doc_string = "\n\n".join(str(chunk) for chunk in retrieved_chunks)
    
    prompt = f"""
You are a strict teacher grading a test. You are given a set of FACTS and a STUDENT ANSWER.
Your goal is to determine if the STUDENT ANSWER is grounded entirely in the FACTS.

FACTS: {doc_string}
STUDENT ANSWER: {actual_answer}

RULES:
1. If the STUDENT ANSWER contains any "hallucinated" information that cannot be directly proven by the FACTS, it is ungrounded.
2. If the STUDENT ANSWER accurately reflects the FACTS, it is grounded.
3. If the FACTS contains nothing or empty, the student answer should say something related to "I dont know".

Return EXACTLY ONE WORD: "True" if grounded, or "False" if it hallucinates.
"""
    cmd = ["gemini", "--model", "gemini-2.5-flash-lite"]
    try:
        result = subprocess.run(
            cmd, 
            input=prompt,
            capture_output=True, 
            text=True, 
            check=True, 
            encoding="utf-8"
        )
        
        return "true" in result.stdout.strip().lower()
    except Exception as e:
        logger.warning(f"Error calling Gemini CLI Judge: {e}")
        return False

def evaluate_correctness_with_gemini(skill_context, ground_truth, actual_answer):
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
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Benchmarking")
    parser.add_argument("--config", type=str, default="Hybrid", choices=["Vector", "Graph", "BM25", "Hybrid"], help="Configuration name to evaluate")
    args = parser.parse_args()

    input_jsonl = f"results/rag_outputs_{args.config}.jsonl"
    output_csv = f"results/evaluation_results_{args.config}.csv"

    logger.info(f"Starting Evaluation flow for config: {args.config}...")

    if not os.path.exists(input_jsonl):
        logger.error(f"Inference data file not found: {input_jsonl}")
        logger.info(f"Please run run_rag_inference.py --config {args.config} first!")
        return

    skill_context = load_evaluator_skill()
    if not skill_context: return

    # --- LOGIC CHECKPOINT ---
    processed_queries = set()
    file_exists = os.path.exists(output_csv)

    if file_exists:
        try:
            existing_df = pd.read_csv(output_csv)
            if 'query' in existing_df.columns:
                processed_queries = set(existing_df['query'].tolist())
                logger.info(f"Found {len(processed_queries)} already scored queries. Resuming remaining ones...")
        except Exception:
            pass

    # Read all data from JSONL
    inference_records = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                inference_records.append(json.loads(line))

    logger.info(f"Starting scoring for {len(inference_records)} records...")

    for record in tqdm(inference_records):
        query = record['query']

        if query in processed_queries:
            continue

        retrieved_chunks = record.get('retrieved_chunks', [])
        actual_answer = str(record['actual_answer'])
        expected_context = record.get('expected_context', '')
        ground_truth = str(record['ground_truth_answer'])

        is_retrieval_good = evaluate_retrieval_with_gemini(query, retrieved_chunks, expected_context)
            
        if is_retrieval_good is None:
            logger.error("System error (possibly API Quota exhausted). Script will pause to preserve data.")
            break

        retrieval_score = 1 if is_retrieval_good else 0
        
        is_grounded = evaluate_groundedness_with_gemini(retrieved_chunks, actual_answer)
        if is_grounded is None:
            logger.error("System error (possibly API Quota exhausted). Script will pause to preserve data.")
            break

        groundedness_score = 1 if is_grounded else 0

        correctness_score = evaluate_correctness_with_gemini(
            skill_context, 
            ground_truth, 
            actual_answer
        )

        # Handle Quota exhausted
        if correctness_score is None:
            logger.error("System error (possibly API Quota exhausted). Script will pause to preserve data.")
            break

        # Package results
        row_result = {
            "query": query,
            "retrieval_score": retrieval_score,
            "groundedness_score": groundedness_score,
            "correctness_score": correctness_score,
            "latency": record['latency']
        }

        # Save checkpoint immediately
        res_df = pd.DataFrame([row_result])
        res_df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        file_exists = True 

        time.sleep(1) # Prevent Rate Limit

    # ==========================================
    # 3. FINAL REPORT SUMMARY
    # ==========================================
    if os.path.exists(output_csv):
        final_df = pd.read_csv(output_csv)
        report = {
            "Avg Retrieval Score": final_df['retrieval_score'].mean(),
            "Avg Groundedness Score": final_df['groundedness_score'].mean(),
            "Avg Correctness Score": final_df['correctness_score'].mean(),
            "Avg Latency (s)": final_df['latency'].mean()
        }

        print("\n" + "="*45)
        print(f"🏆 RAG SYSTEM PERFORMANCE REPORT ({args.config})")
        print("="*45)
        for metric, value in report.items():
            print(f"{metric:30}: {value:.2f}")
        print("="*45)

if __name__ == "__main__":
    main()
