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

# ==========================================
# 1. CẤU HÌNH GIÁM KHẢO (JUDGE LLM)
# ==========================================
judge_llm = ChatOllama(model=config.LLM_MODEL, temperature=0) 

JUDGE_PROMPT = PromptTemplate.from_template("""
Bạn là một giám khảo chuyên gia. Hãy so sánh CÂU TRẢ LỜI THỰC TẾ với CÂU TRẢ LỜI MẪU (Ground Truth).
Chấm điểm từ 0 đến 10 dựa trên:
- Độ chính xác thông tin (Trọng số cao nhất).
- Sự đầy đủ so với ý chính của mẫu.

Chỉ trả về duy nhất một con số từ 0 đến 10. Không giải thích.

--- CÂU TRẢ LỜI MẪU ---
{ground_truth}

--- CÂU TRẢ LỜI THỰC TẾ ---
{actual_answer}

Điểm số:""")

# ==========================================
# 2. KHỞI TẠO HỆ THỐNG
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

print(f"✅ Đã khởi tạo thành công RAGEngine với LLM: {config.LLM_MODEL}")

def run_benchmark(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file benchmark tại {csv_path}")
        return

    print(f"📊 Đang tải tập test từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Giới hạn số câu để test nhanh
    # df = df.sample(min(20, len(df))) 

    results = []
    
    print(f"🔎 Bắt đầu đánh giá {len(df)} câu hỏi...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            start_time = time.time()
            
            result = engine.answer_question(row['query'])

            # A. Đo lường Routing
            actual_intent = result['intent']
            routing_is_correct = 1 if actual_intent.lower() == row['expected_intent'].lower() else 0
            
            # B. Đo lường Retrieval
            context = result["context"]
            retrieval_is_hit = 1 if str(row['expected_context']).lower() in context.lower() else 0
            
            # C. Đo lường Generation
            actual_answer = result["answer"]
            
            # D. LLM-as-a-Judge chấm điểm
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
            print(f"⚠️ Lỗi khi xử lý câu hỏi '{row['query']}': {e}")
            continue

    # ==========================================
    # 4. TỔNG HỢP VÀ XUẤT BÁO CÁO
    # ==========================================
    if not results:
        print("❌ Không có kết quả nào được tạo ra.")
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
    print("🏆 BÁO CÁO HIỆU NĂNG HỆ THỐNG RAG")
    print("="*40)
    for metric, value in report.items():
        print(f"{metric:30}: {value:.2f}")
    print("="*40)
    
    res_df.to_csv("evaluation_results.csv", index=False)
    print("✅ Đã lưu kết quả chi tiết vào evaluation_results.csv")

if __name__ == "__main__":
    # Tự động tìm file benchmark ở root hoặc folder tests
    csv_file = "benchmark_dataset_2000.csv"
    if not os.path.exists(csv_file):
        csv_file = "tests/benchmark_dataset_2000.csv"
        
    run_benchmark(csv_file)
