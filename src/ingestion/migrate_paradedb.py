import psycopg2
from src.core.config import config
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def migrate():
    """
    Enables pg_search extension and creates a BM25 index on the lore chunks.
    """
    # Parse URI for psycopg2
    # config.POSTGRES_URI format: postgresql+psycopg2://user:pass@host:port/db
    connection_uri = config.POSTGRES_URI.replace("postgresql+psycopg2://", "postgresql://")

    try:
        conn = psycopg2.connect(connection_uri)
        conn.autocommit = True
        cur = conn.cursor()

        logger.info("Enabling pg_search extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")

        # We assume the default LangChain table name 'langchain_pg_embedding'
        # and the column name 'document' for the text content.
        table_name = "langchain_pg_embedding"
        
        logger.info(f"Creating BM25 index on {table_name}(document) with English stemming...")
        
        # ParadeDB syntax for custom tokenizer with stemming (v2 API)
        # We use 'id' as the key_field which is the primary key for LangChain PGVector table
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS bm25_doc_idx 
        ON {table_name} 
        USING bm25 (id, (document::pdb.simple('stemmer=english')))
        WITH (key_field='id');
        """
        
        cur.execute(create_index_sql)
        logger.info("Migration successful: pg_search enabled and BM25 index created.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate()
