import csv
import os
    
def main():
    # Lấy danh sách tướng chính xác từ thư mục processed_data
    data_dir = 'processed_data'
    if not os.path.exists(data_dir):
        print(f"Thư mục {data_dir} không tồn tại.")
        return

    champions = [f.replace('.json', '') for f in os.listdir(data_dir) if f.endswith('.json')]
    # Sắp xếp theo chiều dài giảm dần để ưu tiên match tên dài (vd: "Aurelion Sol" sẽ được match trước "Aurelion")
    champions.sort(key=len, reverse=True)

    input_file = "benchmark_dataset_cleann.csv"
    output_file = "benchmark_dataset_small.csv"

    rows_to_keep = []
    champ_counts = {c: 0 for c in champions}

    print("Đang xử lý dữ liệu...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                query = row["query"]
                context = row["expected_context"]
                
                # Tìm tướng tương ứng với câu hỏi này
                found_champ = None
                for champ in champions:
                    # Kiểm tra xem tên tướng có nằm trong context hoặc query không
                    if champ in query or champ in context:
                        found_champ = champ
                        break
                        
                # Nếu tìm thấy tướng và tướng đó chưa có đủ 2 câu hỏi, đưa vào danh sách giữ lại
                if found_champ and champ_counts[found_champ] < 2:
                    rows_to_keep.append(row)
                    champ_counts[found_champ] += 1

        # Ghi ra file CSV mới
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

        print(f"✅ Đã trích xuất thành công {len(rows_to_keep)} câu hỏi ra file {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_file}")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()