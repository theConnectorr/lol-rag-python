from .interfaces import IRetriever, IPromptBuilder, IModelGenerator, IRouter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.core.config import config

# ==========================================
# PLUGIN 1: PROMPT BUILDER
# ==========================================
class StandardPrompt(IPromptBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template("""
        You are an expert in League of Legends lore.
        Answer the question base on this context. If you don't know, just say "I don't know".

        --- Context ---
        {context}

        --- Question ---
        {query}

        Your answer:
        """)

    def build(self, query: str, context: str) -> str:
        return self.template.format(context=context, query=query)

# ==========================================
# PLUGIN 2: MODEL GENERATOR
# ==========================================
class LocalLLMGenerator(IModelGenerator):
    def __init__(self, model_name="gemma3:1b", temperature=0.1):
        # Initialize Local model via Ollama
        self.llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=model_name, 
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# PLUGIN 3: ROUTER (Currently a Mock for flow integration)
# ==========================================
class KeywordRouter(IRouter):
    def __init__(self, key):
        self.key = key

    def route(self, query: str) -> str:
        return self.key
    # def route(self, query: str) -> str:
    #     return "Graph"
        # query_lower = query.lower()

        # # 1. GRAPH INTENT (Queries about entities, clear relationships, listings - Bilingual)
        # graph_keywords = [
        #     "list", "how many", "all", "weapon", "region", 
        #     "homeland", "who is", "mother", "father", "brother", 
        #     "sister", "family", "item", "where is", "belong to", "wield"
        # ]
        # if any(word in query_lower for word in graph_keywords):
        #     return "Graph"

        # # 2. HYBRID INTENT (Cross-inference, relationship analysis + context - Bilingual)
        # hybrid_keywords = [
        #     "why", "relationship", "role", "affect", "effect", 
        #     "compare", "reason", "connection", "interact", "between"
        # ]
        # if any(word in query_lower for word in hybrid_keywords):
        #     return "Hybrid"

        # # 3. VECTOR INTENT (Psychology, history, past, description - Bilingual)
        # vector_keywords = [
        #     "past", "personality", "event", "tell me about", "summary", 
        #     "history", "feel", "lore", "background", "story", "describe"
        # ]
        # if any(word in query_lower for word in vector_keywords):
        #     return "Vector"

        # # 4. Default fallback
        # return "Vector"