import os
import json
from neo4j import GraphDatabase
from src.core.config import config

DATA_DIR = "processed_data/"
ALIAS_FILE = "alias_mapping.json"

def load_alias_mapping():
    try:
        with open(ALIAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ Không tìm thấy alias_mapping.json. Dữ liệu có thể bị phân mảnh.")
        return {}

def main():
    alias_map = load_alias_mapping()
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    # Kết nối Neo4j
    driver = GraphDatabase.driver(
        config.NEO4J_URI, 
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

    print("🚀 Bắt đầu xây dựng Đồ thị Tri thức (Knowledge Graph) trên Neo4j...")

    try:
        with driver.session() as session:
            # Bỏ comment dòng dưới nếu bạn muốn XÓA SẠCH đồ thị cũ trước khi insert
            # session.run("MATCH (n) DETACH DELETE n")

            for filename in files:
                champion_id = filename.replace(".json", "")
                
                with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    
                infobox = file_data.get("infobox", {})
                print(f"[GRAPH] Đang xử lý: {champion_id}...")

                # BƯỚC A: TẠO NODE TƯỚNG (Quy chuẩn bằng Alias Map)
                champ_name = alias_map.get(champion_id.lower(), champion_id)
                session.run(
                    "MERGE (c:Champion {name: $name})", 
                    name=champ_name
                )

                # BƯỚC B: VÙNG ĐẤT
                regions = infobox.get("Place of origin", infobox.get("Region(s)", []))
                for region in regions:
                    if region:
                        session.run("""
                            MATCH (c:Champion {name: $champName})
                            MERGE (r:Region {name: $regionName})
                            MERGE (c)-[:BELONGS_TO]->(r)
                        """, champName=champ_name, regionName=region.strip())

                # BƯỚC C: VŨ KHÍ
                weapons = infobox.get("Weapon(s)", infobox.get("Weapon", []))
                for weapon in weapons:
                    if weapon:
                        session.run("""
                            MATCH (c:Champion {name: $champName})
                            MERGE (w:Weapon {name: $weaponName})
                            MERGE (c)-[:WIELDS]->(w)
                        """, champName=champ_name, weaponName=weapon.strip())

                # BƯỚC D: QUAN HỆ TƯỚNG
                related_chars = infobox.get("Related character", [])
                for raw_name in related_chars:
                    if raw_name:
                        # Dò từ điển để quy chuẩn Node
                        normalized_target = alias_map.get(raw_name.lower().strip(), raw_name.strip())
                        session.run("""
                            MATCH (c:Champion {name: $sourceName})
                            MERGE (t:Champion {name: $targetName})
                            MERGE (c)-[:RELATED_TO]->(t)
                        """, sourceName=champ_name, targetName=normalized_target)

                # BƯỚC E: GLiNER ENTITIES (New Step)
                gliner_entities = file_data.get("gliner_entities", {})
                
                # 1. Organization
                for org in gliner_entities.get("Organization", []):
                    session.run("""
                        MATCH (c:Champion {name: $champName})
                        MERGE (o:Organization {name: $orgName})
                        MERGE (c)-[:BELONGS_TO_ORGANIZATION]->(o)
                    """, champName=champ_name, orgName=org)
                    
                # 2. Title
                for title in gliner_entities.get("Title", []):
                    session.run("""
                        MATCH (c:Champion {name: $champName})
                        MERGE (t:Title {name: $titleName})
                        MERGE (c)-[:HAS_TITLE]->(t)
                    """, champName=champ_name, titleName=title)
                    
                # 3. Family
                for family in gliner_entities.get("Family", []):
                    session.run("""
                        MATCH (c:Champion {name: $champName})
                        MERGE (f:Family {name: $familyName})
                        MERGE (c)-[:HAS_FAMILY_MEMBER]->(f)
                    """, champName=champ_name, familyName=family)

        print("\n✅ Đã chèn xong toàn bộ Dữ liệu vào Đồ thị!")

    except Exception as e:
        print(f"❌ Lỗi khi thao tác với Neo4j: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
