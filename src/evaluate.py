import pandas as pd
import time
import os
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from src.core.config import config
from src.core.engine import RAGEngine
from src.core.plugins import (
    KeywordRouter,
    PostgresVectorRetriever,
    Neo4jGraphRetriever,
    HybridRRFRetriever,
    StandardPrompt,
    LocalLLMGenerator
)
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# ==========================================
# 1. JUDGE LLM CONFIGURATION
# ==========================================
judge_llm = ChatOllama(model=config.LLM_MODEL, temperature=0) 

JUDGE_PROMPT = PromptTemplate.from_template("""
You are an expert judge. Compare the ACTUAL ANSWER with the GROUND TRUTH.
Score from 0 to 10 based on:
- Information accuracy (Highest weight).
- Completeness compared to the main points of the ground truth.

Only return a single number from 0 to 10. Do not explain.

--- GROUND TRUTH ---
{ground_truth}

--- ACTUAL ANSWER ---
{actual_answer}

Score:""")

# ==========================================
# 2. SYSTEM INITIALIZATION
# ==========================================
vector_retriever = PostgresVectorRetriever(
    connection_string=config.POSTGRES_URI,
    collection_name=config.POSTGRES_COLLECTION
)

graph_retriever = Neo4jGraphRetriever(
    uri=config.NEO4J_URI,
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD
)

engine = RAGEngine(
    router=KeywordRouter(),
    retrievers={
        "Vector": vector_retriever,
        "Graph": graph_retriever,
        "Hybrid": HybridRRFRetriever(
            vector_retriever=vector_retriever, 
            graph_retriever=graph_retriever
        )
    },
    prompt_builder=StandardPrompt(),
    generator=LocalLLMGenerator(
        model_name=config.LLM_MODEL, 
        temperature=config.LLM_TEMPERATURE
    )
)

logger.info(f"RAGEngine successfully initialized with LLM: {config.LLM_MODEL}")

def run_benchmark(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"Error: Benchmark file not found at {csv_path}")
        return

    logger.info(f"Loading test set from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Limit number of queries for fast testing
    # df = df.sample(min(20, len(df))) 

    results = []
    
    logger.info(f"Starting evaluation of {len(df)} questions...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            start_time = time.time()
            
            result = engine.answer_question(row['query'])

            # A. Measure Routing
            actual_intent = result['intent']
            routing_is_correct = 1 if actual_intent.lower() == row['expected_intent'].lower() else 0
            
            # B. Measure Retrieval
            context = result["context"]
            retrieval_is_hit = 1 if str(row['expected_context']).lower() in context.lower() else 0
            
            # C. Measure Generation
            actual_answer = result["answer"]
            
            # D. LLM-as-a-Judge scoring
            prompt = JUDGE_PROMPT.format(
                ground_truth=row['ground_truth_answer'],
                actual_answer=actual_answer
            )
            
            score_response = judge_llm.invoke(prompt)
            try:
                gen_score = int(score_response.content.strip())
            except:
                gen_score = 0
                
            latency = time.time() - start_time
            
            results.append({
                "routing_correct": routing_is_correct,
                "retrieval_hit": retrieval_is_hit,
                "gen_score": gen_score,
                "latency": latency
            })
        except Exception as e:
            logger.error(f"Error while processing question '{row['query']}': {e}")
            continue

    # ==========================================
    # 4. SUMMARY AND REPORT EXPORT
    # ==========================================
    if not results:
        logger.error("No results generated.")
        return

    res_df = pd.DataFrame(results)
    
    report = {
        "Routing Accuracy (%)": res_df['routing_correct'].mean() * 100,
        "Retrieval Hit Rate (%)": res_df['retrieval_hit'].mean() * 100,
        "Avg Generation Score (0-10)": res_df['gen_score'].mean(),
        "Avg Latency (s)": res_df['latency'].mean(),
        "Max Latency (s)": res_df['latency'].max()
    }
    
    print("\n" + "="*40)
    print("🏆 RAG SYSTEM PERFORMANCE REPORT")
    print("="*40)
    for metric, value in report.items():
        print(f"{metric:30}: {value:.2f}")
    print("="*40)
    
    res_df.to_csv("evaluation_results.csv", index=False)
    logger.info("Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    # Automatically find benchmark file in root or tests folder
    csv_file = "benchmark_dataset_2000.csv"
    if not os.path.exists(csv_file):
        csv_file = "tests/benchmark_dataset_2000.csv"
        
    run_benchmark(csv_file)
