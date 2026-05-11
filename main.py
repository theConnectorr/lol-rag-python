from gliner import GLiNER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.core.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Loading GLiNER and REBEL models into RAM...")
# 1. Initialize GLiNER
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# 2. Initialize REBEL (Explicitly - No Pipeline)
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def extract_triplets(text):
    # Step 1: Let GLiNER scan for entity shaping
    labels = ["Champion", "Region", "Weapon", "Title", "Organization", "Family"]
    entities = gliner_model.predict_entities(text, labels, threshold=0.5)

    logger.info("GLiNER found Nodes:")
    for ent in entities:
        logger.info(f" - [{ent['label']}] {ent['text']}")

    # Step 2: Feed the entire text into REBEL
    logger.info("REBEL is extracting Edges...")

    # Package text into PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    # Text generation configuration (Optimized for REBEL)
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }

    # Execute model
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # Decode results from tensors back to text
    # Note: skip_special_tokens=False is CRITICALLY IMPORTANT to keep <triplet>, <subj>, <obj> tags
    extracted_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Parse returned text into JSON
    extracted_triplets = extract_relations_from_rebel_output(extracted_text[0])

    for triplet in extracted_triplets:
        logger.info(f" 🎯 {triplet['head']} --[{triplet['type']}]--> {triplet['tail']}")

def extract_relations_from_rebel_output(text):
    # Classic helper function used by the community for REBEL output parsing
    relations = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't': subject += ' ' + token
            elif current == 's': object_ += ' ' + token
            elif current == 'o': relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return relations

# TEST TEXT
test_text = "Garen Crownguard is a proud soldier of Demacia. He wields a massive broadsword called Sunfire and is the brother of Lux."
extract_triplets(test_text)