from abc import ABC, abstractmethod

class IRetriever(ABC):
    """Ổ cắm cho mọi thuật toán Truy xuất dữ liệu (Vector, Graph, BM25)"""
    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass

class IPromptBuilder(ABC):
    """Ổ cắm cho mọi kỹ thuật Prompting (Zero-shot, Few-shot, CoT)"""
    @abstractmethod
    def build(self, query: str, context: str) -> str:
        pass

class IModelGenerator(ABC):
    """Ổ cắm cho mọi Mô hình Sinh văn bản (Gemma, Llama, Gemini API)"""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class IRouter(ABC):
    """Ổ cắm cho Bộ phân tích Truy vấn (Query Router)"""
    @abstractmethod
    def route(self, query: str) -> str:
        """Trả về nhãn luồng: 'Vector', 'Graph', hoặc 'Hybrid'"""
        pass