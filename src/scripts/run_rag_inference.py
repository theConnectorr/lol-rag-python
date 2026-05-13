import os
import json
import time
import pandas as pd
import argparse
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
from src.retrievers.paradedb_keyword_retriever import ParadeDBKeywordRetriever

logger = setup_logger(__name__)

# ==========================================
# 1. INITIALIZE RAG SYSTEM
# ==========================================
def initialize_engine(config_type: str):
    logger.info(f"Initializing RAG Engine for config: {config_type}...")

    # Shared components
    vector_retriever = PostgresVectorRetriever(
        connection_string=config.POSTGRES_URI,
        collection_name=config.POSTGRES_COLLECTION
    )
    
    graph_retriever = Neo4jGraphRetriever(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )
    
    bm25_retriever = ParadeDBKeywordRetriever(
        connection_uri=config.POSTGRES_URI
    )

    # Select retrievers based on config
    retrievers_map = {}
    if config_type == "Vector":
        retrievers_map = {"Vector": vector_retriever}
    elif config_type == "Graph":
        retrievers_map = {"Graph": graph_retriever}
    elif config_type == "BM25":
        retrievers_map = {"BM25": bm25_retriever}
    elif config_type == "Hybrid":
        retrievers_map = {
            "Hybrid": HybridRRFRetriever(
                vector_retriever=vector_retriever,
                graph_retriever=graph_retriever
            )
        }
    else:
        # Fallback to Hybrid
        retrievers_map = {
            "Hybrid": HybridRRFRetriever(vector_retriever, graph_retriever)
        }

    engine = RAGEngine(
        router=KeywordRouter(config_type),
        retrievers=retrievers_map,
        prompt_builder=StandardPrompt(),
        generator=LocalLLMGenerator(
            model_name=config.LLM_MODEL, 
            temperature=config.LLM_TEMPERATURE
        )
    )
    
    logger.info(f"RAGEngine successfully initialized with Generator: {config.LLM_MODEL}")
    return engine

# ==========================================
# 2. INFERENCE FLOW (WITH CHECKPOINT)
# ==========================================
def run_inference(engine, csv_path, output_jsonl):
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
                    "expected_context": str(row['expected_context']),
                    "ground_truth_answer": str(row['ground_truth_answer']),
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
    parser = argparse.ArgumentParser(description="Run RAG Inference Benchmarking")
    parser.add_argument("--config", type=str, default="Hybrid", choices=["Vector", "Graph", "BM25", "Hybrid"], help="Retriever configuration to use")
    parser.add_argument("--input", type=str, default="benchmark_dataset_small.csv", help="Input benchmark CSV file")
    
    args = parser.parse_args()
    
    output_file = f"results/rag_outputs_{args.config}.jsonl"
    os.makedirs("results", exist_ok=True)
    
    engine = initialize_engine(args.config)
    run_inference(engine, args.input, output_file)