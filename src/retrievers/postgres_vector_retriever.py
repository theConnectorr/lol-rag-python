from src.core.interfaces import IRetriever
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from src.core.config import config

class PostgresVectorRetriever(IRetriever):
    def __init__(self, connection_string: str, collection_name: str = "lore_chunks"):
        self.embeddings = OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL
        )
        
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> str:
        # Thực hiện Cosine Similarity Search
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        # Gom kết quả thành 1 cục Text để mớm cho LLM
        return "\n\n".join([doc.page_content for doc in docs])