import os

DATA_DIR = "processed_data/"
INPUT_FILE = "benchmark_dataset_gemini.csv"
OUTPUT_FILE = "benchmark_dataset_clean.csv"

def main():
    print("🧹 Bắt đầu dọn dẹp file CSV...")

    # 1. Quét thư mục data để lấy danh sách toàn bộ Tướng hợp lệ
    try:
        valid_champs = [f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy thư mục {DATA_DIR}.")
        return

    print(f"📦 Đã load {len(valid_champs)} tướng làm bộ lọc chuẩn.")

    valid_lines = []
    header_found = False
    garbage_count = 0

    # 2. Đọc từng dòng của file CSV lỗi
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # Bỏ qua các dòng trống hoàn toàn
        if not line:
            continue

        # Xử lý dòng Header: Chỉ giữ lại 1 dòng header duy nhất
        if line.lower().startswith("champion_name"):
            if not header_found:
                valid_lines.append(line)
                header_found = True
            continue

        # 3. Kiểm tra xem dòng này có bắt đầu bằng tên tướng hợp lệ hay không
        is_valid = False
        for champ in valid_champs:
            # LLM có thể xuất ra: Lee Sin,"Câu hỏi..." hoặc "Lee Sin","Câu hỏi..."
            if line.startswith(f'{champ},') or line.startswith(f'"{champ}",'):
                is_valid = True
                break

        if is_valid:
            valid_lines.append(line)
        else:
            garbage_count += 1
            # Bạn có thể in ra để xem script đã lọc đi những gì
            # print(f"🗑️ Đã xóa: {line[:50]}...") 

    # 4. Ghi dữ liệu sạch ra file mới
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for clean_line in valid_lines:
            f.write(clean_line + "\n")

    print("\n===========================================")
    print(f"✅ Dọn dẹp hoàn tất!")
    print(f"✅ Đã giữ lại: {len(valid_lines) - 1} dòng dữ liệu chuẩn (không tính header).")
    print(f"🗑️ Đã lọc bỏ: {garbage_count} dòng rác (LLM chat nhảm).")
    print(f"📁 File sạch được lưu tại: {OUTPUT_FILE}")
    print("===========================================")

if __name__ == "__main__":
    main()