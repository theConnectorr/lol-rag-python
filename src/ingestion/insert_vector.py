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

DATA_DIR = "processed_data/"

def main():
    print(f"🚀 Bắt đầu quá trình Embedding bằng {config.EMBEDDING_MODEL}...")

    # 1. Khởi tạo PGVector & Embeddings
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

    # 2. Cấu hình dao cắt (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    # Lấy danh sách tướng từ thư mục
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    for filename in files:
        champion_name = filename.replace(".json", "")
        print(f"\n⚡ Đang xử lý tướng: {champion_name}")

        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Trích xuất metadata
        infobox = data.get("infobox", {})
        place_of_origin = infobox.get("Place of origin", ["Unknown"])[0]

        flat_sections = flatten_toc(data.get("mainContent", []))
        raw_docs = []

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

        # Cắt Chunk
        raw_chunks = text_splitter.split_documents(raw_docs)

        # 🌟 Bộ lọc bọc thép
        docs = []
        for doc in raw_chunks:
            text = doc.page_content
            # Kiểm tra chứa chữ cái/số và độ dài > 10
            if re.search(r'[a-zA-Z0-9]', text) and len(text.strip()) > 10:
                docs.append(doc)

        print(f"✂️ Đã cắt thành {len(docs)} chunks sạch.")

        # 3. Đẩy vào Postgres theo từng Batch
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                print(f"   ✅ Đã insert batch {i // batch_size + 1} / {(len(docs) + batch_size - 1) // batch_size}")
                time.sleep(0.1) # Giảm tải CPU
            except Exception as e:
                print(f"   ❌ Lỗi DB ở batch này: {e}")

    print("\n🎉 Hoàn tất quá trình nhúng Vector vào Postgres!")

if __name__ == "__main__":
    main()