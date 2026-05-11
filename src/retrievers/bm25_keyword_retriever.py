from src.core.interfaces import IRetriever
from langchain_community.retrievers import BM25Retriever
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class BM25KeywordRetriever(IRetriever):
    def __init__(self, raw_documents: list[str]):
        logger.info("Initializing BM25KeywordRetriever...")
        self.retriever = BM25Retriever.from_texts(raw_documents)
        self.retriever.k = 3

    def retrieve(self, query: str) -> list[str]:
        # Changed type hint to list[str]
        logger.debug(f"Retrieving with BM25 for query: {query}")
        docs = self.retriever.invoke(query)
        # Returns the original list of text chunks
        return [doc.page_content for doc in docs]