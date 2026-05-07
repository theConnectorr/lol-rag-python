from .interfaces import IRouter, IRetriever, IPromptBuilder, IModelGenerator

class RAGEngine:
    def __init__(self, 
                 router: IRouter,
                 retrievers: dict[str, IRetriever], # Chứa nhiều phích cắm Retriever
                 prompt_builder: IPromptBuilder, 
                 generator: IModelGenerator):
        self.router = router
        self.retrievers = retrievers
        self.prompt_builder = prompt_builder
        self.generator = generator

    def answer_question(self, query: str) -> dict:
        # Bước 1: Routing (Định tuyến)
        intent = self.router.route(query)
        
        # Bước 2: Chọn Phích cắm Retriever phù hợp (Fallback về Vector nếu lỗi)
        active_retriever = self.retrievers.get(intent, self.retrievers.get("Vector"))
        
        # Bước 3: Móc dữ liệu
        context = active_retriever.retrieve(query)
        
        # Bước 4: Đóng gói & Sinh câu trả lời
        prompt = self.prompt_builder.build(query, context)
        answer = self.generator.generate(prompt)
        
        # TRẢ VỀ TOÀN BỘ METADATA ĐỂ PHỤC VỤ LUỒNG EVALUATE!
        return {
            "query": query,
            "intent": intent,           # Đã bắt được Intent!
            "active_route": type(active_retriever).__name__, 
            "context": context,
            "answer": answer
        }