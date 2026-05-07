from gliner import GLiNER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("⏳ Đang tải mô hình GLiNER và REBEL vào RAM...")
# 1. Khởi tạo GLiNER
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# 2. Khởi tạo REBEL (Cách Tường minh - Không xài Pipeline)
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def extract_triplets(text):
    # Bước 1: Cho GLiNER quét qua để định hình thực thể
    labels = ["Champion", "Region", "Weapon", "Title", "Organization", "Family"]
    entities = gliner_model.predict_entities(text, labels, threshold=0.5)
    
    print("\n🔍 GLiNER tìm thấy các Nút (Nodes):")
    for ent in entities:
        print(f" - [{ent['label']}] {ent['text']}")

    # Bước 2: Đưa nguyên đoạn text vào REBEL
    print("\n🔗 REBEL đang nhổ Quan hệ (Edges)...")
    
    # Đóng gói text thành tensor cho PyTorch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    # Cấu hình sinh text (Tối ưu cho REBEL)
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }
    
    # Thực thi model
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )
    
    # Dịch kết quả từ dạng số (tensors) về lại chữ (text)
    # Lưu ý: skip_special_tokens=False là CỰC KỲ QUAN TRỌNG để giữ lại các tag <triplet>, <subj>, <obj>
    extracted_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    
    # Phân tích text trả về thành JSON
    extracted_triplets = extract_relations_from_rebel_output(extracted_text[0])
    
    for triplet in extracted_triplets:
        print(f" 🎯 {triplet['head']} --[{triplet['type']}]--> {triplet['tail']}")

def extract_relations_from_rebel_output(text):
    # Hàm helper kinh điển của cộng đồng dùng REBEL để bóc tách output
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

# TEXT TEST THỬ NGHIỆM
test_text = "Garen Crownguard is a proud soldier of Demacia. He wields a massive broadsword called Sunfire and is the brother of Lux."
extract_triplets(test_text)