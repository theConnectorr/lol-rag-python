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
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# ==========================================
# TERMINAL COLOR CONFIGURATION (ANSI COLORS)
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
# SYSTEM INITIALIZATION
# ==========================================
def initialize_engine():
    logger.info(f"Starting RAG Engine (Model: {config.LLM_MODEL})...")
    print(f"{Colors.CYAN}⏳ Starting RAG Engine (Model: {config.LLM_MODEL})...{Colors.ENDC}")

    try:
        # If you have already ingested data into Postgres/Neo4j using Python, uncomment these lines:
        vector_retriever = PostgresVectorRetriever(
            config.POSTGRES_URI, 
            config.POSTGRES_COLLECTION
        )

        graph_retriever = Neo4jGraphRetriever(
            config.NEO4J_URI, 
            config.NEO4J_USER, 
            config.NEO4J_PASSWORD
        )

        engine = RAGEngine(
            router=KeywordRouter(),
            retrievers={
                # "Vector": vector_retriever,
                # "Graph": graph_retriever,
                "Hybrid": HybridRRFRetriever(vector_retriever, graph_retriever)
            },
            prompt_builder=StandardPrompt(),
            generator=LocalLLMGenerator(
                model_name=config.LLM_MODEL, 
                temperature=config.LLM_TEMPERATURE
            )
        )

        logger.info("Startup successful! Connected to LLM & Databases.")
        print(f"{Colors.GREEN}✅ Startup successful! Connected to LLM & Databases.{Colors.ENDC}\n")
        return engine
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        print(f"{Colors.RED}❌ Initialization error: {e}{Colors.ENDC}")
        exit(1)

# ==========================================
# INTERACTIVE CHAT LOOP
# ==========================================
def main():
    engine = initialize_engine()

    print(f"{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}🗡️  LEAGUE OF LEGENDS - HYBRID RAG CHATBOT  🗡️{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"Type {Colors.RED}'quit'{Colors.ENDC} or {Colors.RED}'exit'{Colors.ENDC} to exit.\n")

    while True:
        try:
            # 1. Get user input
            user_input = input(f"{Colors.GREEN}{Colors.BOLD}👤 You: {Colors.ENDC}")

            if user_input.lower() in ['quit', 'exit']:
                print(f"\n{Colors.CYAN}Goodbye! See you on Summoner's Rift.{Colors.ENDC}")
                break

            if not user_input.strip():
                continue

            # 2. Process through RAGEngine
            start_time = time.time()
            print(f"{Colors.YELLOW}System is thinking...{Colors.ENDC}")

            result = engine.answer_question(user_input)

            latency = time.time() - start_time

            # 3. Print Metadata (Traceability)
            # print(f"{Colors.CYAN}   ↳ 🔀 Routing (Intent): {Colors.BOLD}{result['intent']}{Colors.ENDC}")
            # print(f"{Colors.CYAN}   ↳ Retriever: {result['active_route']}{Colors.ENDC}")
            print(f"{Colors.CYAN}   ↳ Response Time: {latency:.2f} seconds{Colors.ENDC}")

            # Extract first 150 characters of Context for debug
            # context_preview = result['context'].replace('\n', ' ')[:150] + "..."
            context_preview = result['context'].replace('\n', ' ')
            print(f"{Colors.CYAN}   ↳ Found Context: {context_preview}{Colors.ENDC}\n")

            # 4. Print final answer
            print(f"{Colors.HEADER}{Colors.BOLD}RAG Bot: {Colors.ENDC}{result['answer']}")
            print(f"\n{Colors.BLUE}----------------------------------------------------{Colors.ENDC}\n")

        except KeyboardInterrupt:
            # Handle Ctrl+C
            print(f"\n\n{Colors.CYAN}Goodbye!{Colors.ENDC}")
            break
        except Exception as e:
            logger.error(f"Error during chat loop: {e}", exc_info=True)
            print(f"{Colors.RED}\nAn error occurred: {e}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()