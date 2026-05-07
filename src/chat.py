import time
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

# ==========================================
# CẤU HÌNH MÀU SẮC TERMINAL (ANSI COLORS)
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ==========================================
# KHỞI TẠO HỆ THỐNG
# ==========================================
def initialize_engine():
    print(f"{Colors.CYAN}⏳ Đang khởi động RAG Engine (Model: {config.LLM_MODEL})...{Colors.ENDC}")
    
    try:
        # Nếu bạn đã bơm dữ liệu vào Postgres/Neo4j bằng Python rồi thì mở comment 2 dòng dưới:
        vector_retriever = PostgresVectorRetriever(config.POSTGRES_URI, config.POSTGRES_COLLECTION)
        graph_retriever = Neo4jGraphRetriever(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
        
        # TẠM THỜI DÙNG DUMMY RETRIEVER NẾU CHƯA CÓ DATA MỚI
        # vector_retriever = DummyVectorRetriever()
        # graph_retriever = DummyVectorRetriever()

        engine = RAGEngine(
            router=KeywordRouter(),
            retrievers={
                "Vector": vector_retriever,
                "Graph": graph_retriever,
                "Hybrid": HybridRRFRetriever(vector_retriever, graph_retriever)
            },
            prompt_builder=StandardPrompt(),
            generator=LocalLLMGenerator(
                model_name=config.LLM_MODEL, 
                temperature=config.LLM_TEMPERATURE
            )
        )
        print(f"{Colors.GREEN}✅ Khởi động thành công! Đã kết nối LLM & Cơ sở dữ liệu.{Colors.ENDC}\n")
        return engine
    except Exception as e:
        print(f"{Colors.RED}❌ Lỗi khởi tạo: {e}{Colors.ENDC}")
        exit(1)

# ==========================================
# VÒNG LẶP CHAT (INTERACTIVE LOOP)
# ==========================================
def main():
    engine = initialize_engine()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}🗡️  LEAGUE OF LEGENDS - HYBRID RAG CHATBOT  🗡️{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"Gõ {Colors.RED}'quit'{Colors.ENDC} hoặc {Colors.RED}'exit'{Colors.ENDC} để thoát.\n")

    while True:
        try:
            # 1. Nhận câu hỏi từ user
            user_input = input(f"{Colors.GREEN}{Colors.BOLD}👤 Bạn: {Colors.ENDC}")
            
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n{Colors.CYAN}👋 Tạm biệt! Hẹn gặp lại trên Summoner's Rift.{Colors.ENDC}")
                break
                
            if not user_input.strip():
                continue

            # 2. Xử lý qua RAGEngine
            start_time = time.time()
            print(f"{Colors.YELLOW}🧠 Hệ thống đang suy nghĩ...{Colors.ENDC}")
            
            result = engine.answer_question(user_input)
            
            latency = time.time() - start_time

            # 3. In Metadata (Dấu vết hoạt động)
            print(f"{Colors.CYAN}   ↳ 🔀 Định tuyến (Intent): {Colors.BOLD}{result['intent']}{Colors.ENDC}")
            print(f"{Colors.CYAN}   ↳ 📚 Phích cắm (Retriever): {result['active_route']}{Colors.ENDC}")
            print(f"{Colors.CYAN}   ↳ ⏱️ Thời gian phản hồi: {latency:.2f} giây{Colors.ENDC}")
            
            # Trích xuất 150 ký tự đầu của Context để debug
            context_preview = result['context'].replace('\n', ' ')[:150] + "..."
            print(f"{Colors.CYAN}   ↳ 📝 Ngữ cảnh tìm được: {context_preview}{Colors.ENDC}\n")

            # 4. In câu trả lời cuối cùng
            print(f"{Colors.HEADER}{Colors.BOLD}🤖 RAG Bot: {Colors.ENDC}{result['answer']}")
            print(f"\n{Colors.BLUE}----------------------------------------------------{Colors.ENDC}\n")

        except KeyboardInterrupt:
            # Xử lý khi user bấm Ctrl+C
            print(f"\n\n{Colors.CYAN}👋 Tạm biệt!{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.RED}\n❌ Đã xảy ra lỗi: {e}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()