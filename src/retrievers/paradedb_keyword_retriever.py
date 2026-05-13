import psycopg2
from src.core.interfaces import IRetriever
from src.core.config import config
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class ParadeDBKeywordRetriever(IRetriever):
    def __init__(self, connection_uri: str, top_k: int = 5):
        logger.info("Initializing ParadeDBKeywordRetriever...")
        # Convert SQLAlchemy URI to psycopg2-friendly URI
        self.connection_uri = connection_uri.replace("postgresql+psycopg2://", "postgresql://")
        self.top_k = top_k
        self.table_name = "langchain_pg_embedding"

    def retrieve(self, query: str) -> list[str]:
        logger.debug(f"Retrieving with ParadeDB BM25 for query: {query}")

        # Sanitize query: Remove special characters that can break the ParadeDB parser
        # especially single quotes, parentheses, and other Lucene-like symbols
        sanitized_query = query.replace("'", "").replace("?", "").replace("(", "").replace(")", "").replace(":", "")

        try:
            conn = psycopg2.connect(self.connection_uri)
            cur = conn.cursor()

            # Using ParadeDB BM25 '|||' operator which is an OR search across all terms
            # It is more forgiving than '@@@' for natural language queries
            search_sql = f"""
            SELECT document 
            FROM {self.table_name} 
            WHERE document ||| %s
            ORDER BY pdb.score(id) DESC
            LIMIT %s;
            """

            cur.execute(search_sql, (sanitized_query, self.top_k))            
            results = cur.fetchall()
            
            # Extract the 'document' string from each result tuple
            chunks = [row[0] for row in results]
            
            logger.info(f"Retrieved {len(chunks)} chunks using ParadeDB BM25")
            return chunks

        except Exception as e:
            logger.error(f"ParadeDB retrieval failed: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
