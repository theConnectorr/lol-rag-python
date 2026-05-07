import os
from dotenv import load_dotenv

# Tự động tìm và load file .env ở thư mục gốc
load_dotenv()

class Settings:
    """Class chứa toàn bộ cấu hình của hệ thống"""
    
    # --- Neo4j Settings ---
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secretpassword")

    # --- Postgres Settings ---
    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")
    POSTGRES_COLLECTION = os.getenv("POSTGRES_COLLECTION", "lore_chunks")

    # --- LLM & Embedding Settings ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Khởi tạo một object (Singleton) để dùng chung cho toàn dự án
config = Settings()