from src.core.interfaces import IRetriever

class HybridRRFRetriever(IRetriever):
    """
    Hiện tại đóng vai trò là Hybrid Concatenator (Gộp kết quả).
    Trong tương lai sẽ nâng cấp lên RRF chuẩn khi các Retriever trả về List[Document].
    """
    def __init__(self, vector_retriever: IRetriever, graph_retriever: IRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str) -> str:
        # 1. Lấy kết quả từ Vector
        vector_context = self.vector_retriever.retrieve(query)

        # 2. Lấy kết quả từ Graph
        graph_context = self.graph_retriever.retrieve(query)

        # 3. Kết hợp lại
        combined_context = f"--- KẾT QUẢ TỪ VECTOR DB ---\n{vector_context}\n\n"
        combined_context += f"--- KẾT QUẢ TỪ ĐỒ THỊ (NEO4J) ---\n{graph_context}"

        return combined_context