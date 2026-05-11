import os
import json
from gliner import GLiNER
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.text_utils import flatten_toc
from src.core.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = "processed_data/"
LABELS = ["Champion", "Region", "Weapon", "Title", "Organization", "Family"]

def main():
    logger.info("Loading GLiNER model into RAM...")
    gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # Configure text splitter to avoid Truncation warnings (Max 384 tokens ~ 1200 characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"Extracting entities for: {filename}")
        flat_sections = flatten_toc(data.get("mainContent", []))
        
        extracted_entities = {label: set() for label in LABELS}
        
        for section in flat_sections:
            text = section["text"]
            if len(text.strip()) > 10:
                # Split text if it exceeds GLiNER's context window
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
            
    logger.info("Entity extraction and JSON file updates complete!")

if __name__ == "__main__":
    main()
