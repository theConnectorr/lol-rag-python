import os
import json
import re
import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from src.core.config import config
from src.core.text_utils import flatten_toc
from src.core.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = "processed_data/"

def serialize_infobox(champion_name: str, infobox: dict) -> str:
    """
    Transforms a messy Infobox dictionary into a clear text snippet
    so that Vector DB can understand the semantics.
    """
    if not infobox:
        return f"There's no general info for {champion_name}."

    lines = [f"Below is the general info of {champion_name}:"]

    for key, values in infobox.items():
        if values: # Only process if data exists
            # Join array values into a string (e.g., ["Darkin", "Human"] -> "Darkin, Human")
            val_str = ", ".join([str(v) for v in values])
            lines.append(f"- {key}: {val_str}")

    return "\n".join(lines)

def main():
    logger.info(f"Starting Embedding process using {config.EMBEDDING_MODEL}...")

    # 1. Initialize PGVector & Embeddings
    embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_BASE_URL,
        model=config.EMBEDDING_MODEL,
        temperature=0.2
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=config.POSTGRES_COLLECTION,
        connection=config.POSTGRES_URI,
        use_jsonb=True
    )

    # 2. Configure Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    # Get list of champions from the directory
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    for filename in files:
        champion_name = filename.replace(".json", "")
        logger.info(f"Processing champion: {champion_name}")

        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract metadata
        infobox = data.get("infobox", {})
        place_of_origin = infobox.get("Place of origin", ["Unknown"])[0]

        flat_sections = flatten_toc(data.get("mainContent", []))
        raw_docs = []

        infobox_text = serialize_infobox(champion_name, infobox)
        infobox_doc = Document(
            page_content=infobox_text,
            metadata={
                "champion_name": champion_name,
                "section_title": "Infobox / Summary", # Marker for summary chunk
                "region": place_of_origin
            }
        )
        raw_docs.append(infobox_doc) # Insert this chunk first

        for section in flat_sections:
            doc = Document(
                page_content=section["text"],
                metadata={
                    "champion_name": champion_name,
                    "section_title": section["section"],
                    "region": place_of_origin
                }
            )
            raw_docs.append(doc)

        # Chunk the documents
        raw_chunks = text_splitter.split_documents(raw_docs)

        # Clean chunks
        docs = []
        for doc in raw_chunks:
            text = doc.page_content
            # Check for alphanumeric characters and length > 10
            if re.search(r'[a-zA-Z0-9]', text) and len(text.strip()) > 10:
                docs.append(doc)

        logger.info(f"Split into {len(docs)} clean chunks.")

        # 3. Push to Postgres in Batches
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                logger.info(f"   Inserted batch {i // batch_size + 1} / {(len(docs) + batch_size - 1) // batch_size}")
                time.sleep(0.1) # Reduce CPU load
            except Exception as e:
                logger.error(f"   Database error at this batch: {e}")

    logger.info("Vector embedding process into Postgres complete!")

if __name__ == "__main__":
    main()