from src.core.interfaces import IRetriever
from langchain_community import BM25Retriever

class BM25KeywordRetriever(IRetriever):
    def __init__(self, raw_documents: list[str]):
        # Khởi tạo thuật toán Sparse (Tạo chỉ mục từ khóa ngay trên RAM)
        self.retriever = BM25Retriever.from_texts(raw_documents)
        self.retriever.k = 3 # Lấy top 3

    def retrieve(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])