from src.core.interfaces import IRetriever
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class HybridRRFRetriever(IRetriever):
    """
    Currently acting as a Hybrid Concatenator (Merging results).
    In the future, will be upgraded to standard RRF when Retrievers return List[Document].
    """
    def __init__(self, vector_retriever: IRetriever, graph_retriever: IRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str) -> list[str]:
        # 1. Get results from Vector
        logger.debug(f"Hybrid retrieval: fetching vector results for query: {query}")
        vector_context = self.vector_retriever.retrieve(query)

        # 2. Get results from Graph
        logger.debug(f"Hybrid retrieval: fetching graph results for query: {query}")
        graph_context = self.graph_retriever.retrieve(query)

        # 3. Combine them
        logger.info(f"Hybrid retrieval complete: {len(vector_context)} vector chunks, {len(graph_context)} graph chunks.")
        
        return vector_context + graph_context