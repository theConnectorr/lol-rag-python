from .interfaces import IRetriever, IPromptBuilder, IModelGenerator, IRouter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.core.config import config

# ==========================================
# PHÍCH CẮM 1: PROMPT BUILDER
# ==========================================
class StandardPrompt(IPromptBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template("""
        Bạn là chuyên gia về cốt truyện Liên Minh Huyền Thoại.
        Trả lời câu hỏi dựa trên NGỮ CẢNH dưới đây. Nếu không biết, hãy nói không biết.
        
        --- NGỮ CẢNH ---
        {context}
        
        --- CÂU HỎI ---
        {query}
        
        Câu trả lời:
        """)

    def build(self, query: str, context: str) -> str:
        return self.template.format(context=context, query=query)

# ==========================================
# PHÍCH CẮM 2: MODEL GENERATOR
# ==========================================
class LocalLLMGenerator(IModelGenerator):
    def __init__(self, model_name="gemma3:1b", temperature=0.1):
        # Khởi tạo mô hình Local qua Ollama
        self.llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=model_name, 
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# PHÍCH CẮM 3: RETRIEVER (Tạm thời là Mock để ráp luồng)
# ==========================================
class DummyVectorRetriever(IRetriever):
    """Phích cắm tạm thời để test kiến trúc trước khi nối DB thật"""
    def retrieve(self, query: str) -> str:
        # Tương lai: Chỗ này sẽ gọi pgvector hoặc FAISS
        return f"[Dữ liệu giả lập lấy từ VectorDB cho câu hỏi: {query}] - Garen là anh trai của Lux."
    
class KeywordRouter(IRouter):
    def route(self, query: str) -> str:
        query_lower = query.lower()
        # Nếu có chữ liệt kê, đếm -> Dùng Graph
        if any(word in query_lower for word in ["liệt kê", "bao nhiêu", "tất cả"]):
            return "Graph"
        # Nếu có từ khóa phân tích -> Dùng Hybrid
        elif any(word in query_lower for word in ["tại sao", "mối quan hệ"]):
            return "Hybrid"
        # Mặc định -> Vector
        return "Vector"