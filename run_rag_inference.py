import os
import json
import time
import pandas as pd
from tqdm import tqdm
from src.core.logger import setup_logger

from src.core.config import config
from src.core.engine import RAGEngine

from src.core.plugins import (
    KeywordRouter,
    StandardPrompt,
    LocalLLMGenerator
)

from src.retrievers.postgres_vector_retriever import PostgresVectorRetriever
from src.retrievers.neo4j_graph_retriever import Neo4jGraphRetriever
from src.retrievers.hybrid_rrf_retriever import HybridRRFRetriever

logger = setup_logger(__name__)

# ==========================================
# 1. INITIALIZE RAG SYSTEM
# ==========================================
logger.info("Initializing RAG Engine...")

# Initialize Retrievers (Adjust class names if using BM25/Milvus)
vector_retriever = PostgresVectorRetriever(
    connection_string=config.POSTGRES_URI,
    collection_name=config.POSTGRES_COLLECTION
)

# graph_retriever = Neo4jGraphRetriever(
#     uri=config.NEO4J_URI,
#     user=config.NEO4J_USER,
#     password=config.NEO4J_PASSWORD
# )

engine = RAGEngine(
    router=KeywordRouter(),
    retrievers={
        "Vector": vector_retriever,
        # "Graph": graph_retriever,
        # "Hybrid": HybridRRFRetriever(
        #     vector_retriever=vector_retriever, 
        #     graph_retriever=graph_retriever
        # )
    },
    prompt_builder=StandardPrompt(),
    generator=LocalLLMGenerator(
        model_name=config.LLM_MODEL, 
        temperature=config.LLM_TEMPERATURE
    )
)

logger.info(f"RAGEngine successfully initialized with Generator: {config.LLM_MODEL}")

# ==========================================
# 2. INFERENCE FLOW (WITH CHECKPOINT)
# ==========================================
def run_inference(csv_path, output_jsonl="rag_outputs.jsonl"):
    if not os.path.exists(csv_path):
        logger.error(f"Error: Benchmark file not found at {csv_path}")
        return

    logger.info(f"Loading test set from: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    
    # --- RESUME LOGIC (CHECKPOINT) ---
    processed_queries = set()
    
    if os.path.exists(output_jsonl):
        try:
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_queries.add(data.get("query"))
            logger.info(f"⏩ Found existing results file. Skipping {len(processed_queries)} already processed queries.")
        except Exception as e:
            logger.error(f"⚠️ Error reading existing results file: {e}")

    logger.info("Starting Inference for the remaining questions...")
    
    # Open file in append mode ('a')
    with open(output_jsonl, 'a', encoding='utf-8') as out_file:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            query = row['query']
            
            if query in processed_queries:
                continue
                
            try:
                start_time = time.time()
                
                # Call RAG Engine (Returns dict with intent, chunks, answer...)
                result = engine.answer_question(query)
                latency = time.time() - start_time
                
                # Package actual data + expected data into one Record
                record = {
                    "query": query,
                    "expected_intent": str(row['expected_intent']),
                    "expected_context": str(row['expected_context']),
                    "ground_truth_answer": str(row['ground_truth_answer']),
                    "actual_intent": result.get("intent", "Unknown"),
                    "active_route": result.get("active_route", "Unknown"),
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "actual_answer": result.get("answer", ""),
                    "latency": round(latency, 2)
                }
                
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                
            except Exception as e:
                logger.error(f"Error while processing question '{query}': {e}", exc_info=True)
                continue

    logger.info(f"Inference complete! Raw data exported to: {output_jsonl}")

if __name__ == "__main__":
    csv_file = "benchmark_dataset_clean.csv" 
        
    run_inference(csv_file)