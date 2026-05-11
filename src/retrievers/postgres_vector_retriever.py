from src.core.interfaces import IRetriever
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from src.core.config import config
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class PostgresVectorRetriever(IRetriever):
    def __init__(self, connection_string: str, collection_name: str = "lore_chunks"):
        logger.info(f"Initializing PostgresVectorRetriever with collection: {collection_name}")
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

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        # Perform Cosine Similarity Search
        logger.debug(f"Retrieving with PGVector for query: {query}")
        results = self.vector_store.similarity_search(query, k=top_k)
        
        return [doc.page_content for doc in results]