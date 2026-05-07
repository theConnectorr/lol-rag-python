from src.core.interfaces import IRetriever
from neo4j import GraphDatabase

class Neo4jGraphRetriever(IRetriever):
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def retrieve(self, query: str) -> str:
        # Tương lai: Dùng LLM/GLiNER để bóc tên Tướng từ 'query'
        # Hiện tại mock: Bóc đại từ đầu tiên làm keyword để test luồng
        target_entity = query.split()[0] 
        
        # Câu lệnh quét đồ thị tìm mối quan hệ
        cypher_query = """
        MATCH (c:Champion)
        WHERE toLower(c.name) CONTAINS toLower($keyword)
        OPTIONAL MATCH (c)-[r]->(t)
        RETURN c.name AS source, type(r) AS relation, t.name AS target LIMIT 10
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, keyword=target_entity)
            records = [f"{rec['source']} --[{rec['relation']}]--> {rec['target']}" for rec in result]
            
        if not records:
            return "Không tìm thấy dữ liệu liên quan trong Đồ thị."
            
        return "Dữ liệu đồ thị:\n" + "\n".join(records)