from .interfaces import IRouter, IRetriever, IPromptBuilder, IModelGenerator
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class RAGEngine:
    def __init__(self, 
                 router: IRouter,
                 retrievers: dict[str, IRetriever], # Contains multiple Retriever plugins
                 prompt_builder: IPromptBuilder, 
                 generator: IModelGenerator):
        self.router = router
        self.retrievers = retrievers
        self.prompt_builder = prompt_builder
        self.generator = generator

    def answer_question(self, query: str) -> dict:
        # Step 1: Routing
        intent = self.router.route(query)
        logger.debug(f"Routed intent: {intent}")

        # Step 2: Choose the appropriate Retriever plugin
        active_retriever = self.retrievers.get(intent, self.retrievers.get("Vector"))

        # Step 3: Retrieve data - Returns a List[str]
        raw_chunks = active_retriever.retrieve(query)
        logger.info(f"Retrieved {len(raw_chunks)} chunks using {type(active_retriever).__name__}")

        # Merge chunks into a single string for the prompt
        merged_context = "\n---\n".join(raw_chunks)

        # Step 4: Package & Generate Answer
        prompt = self.prompt_builder.build(query, merged_context)
        answer = self.generator.generate(prompt)

        # Return all metadata to serve the evaluation flow
        return {
            "query": query,
            "intent": intent,           
            "active_route": type(active_retriever).__name__, 
            "context": merged_context,          # Merged string (for Prompt/Debug)
            "retrieved_chunks": raw_chunks,     # Crucial for calculating Recall/Precision
            "answer": answer
        }