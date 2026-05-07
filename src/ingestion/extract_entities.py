import os
import json
from gliner import GLiNER
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.text_utils import flatten_toc

DATA_DIR = "processed_data/"
LABELS = ["Champion", "Region", "Weapon", "Title", "Organization", "Family"]

def main():
    print("⏳ Đang tải mô hình GLiNER vào RAM...")
    gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # Cấu hình bộ cắt text để tránh cảnh báo Truncation (Max 384 tokens ~ 1200 characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"🔍 Đang trích xuất thực thể cho: {filename}")
        flat_sections = flatten_toc(data.get("mainContent", []))
        
        extracted_entities = {label: set() for label in LABELS}
        
        for section in flat_sections:
            text = section["text"]
            if len(text.strip()) > 10:
                # Chia nhỏ đoạn văn nếu nó quá dài so với context window của GLiNER
                sub_chunks = text_splitter.split_text(text)
                
                for chunk in sub_chunks:
                    # Predict entities for the sub-chunk
                    entities = gliner_model.predict_entities(chunk, LABELS, threshold=0.5)
                    for ent in entities:
                        extracted_entities[ent["label"]].add(ent["text"].strip())
                    
        # Convert sets to lists for JSON serialization
        data["gliner_entities"] = {k: list(v) for k, v in extracted_entities.items()}
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    print("✅ Hoàn tất trích xuất và cập nhật file JSON!")

if __name__ == "__main__":
    main()
