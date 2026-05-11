from abc import ABC, abstractmethod

class IRetriever(ABC):
    """Interface for all data retrieval algorithms (Vector, Graph, BM25)"""
    @abstractmethod
    def retrieve(self, query: str) -> list[str]:
        pass

class IPromptBuilder(ABC):
    """Interface for all prompting techniques (Zero-shot, Few-shot, CoT)"""
    @abstractmethod
    def build(self, query: str, context: str) -> str:
        pass

class IModelGenerator(ABC):
    """Interface for all text generation models (Gemma, Llama, Gemini API)"""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class IRouter(ABC):
    """Interface for Query Analysis (Query Router)"""
    @abstractmethod
    def route(self, query: str) -> str:
        """Returns the flow label: 'Vector', 'Graph', or 'Hybrid'"""
        pass