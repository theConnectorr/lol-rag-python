# from src.core.interfaces import IRetriever
# from neo4j import GraphDatabase

# class Neo4jGraphRetriever(IRetriever):
#     def __init__(self, uri, user, password):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))

#     def retrieve(self, query: str) -> list[str]:
#         target_entity = query.split()[0]
        
#         cypher_query = """
#             MATCH (c:Champion)
#             WHERE toLower(c.name) CONTAINS toLower($keyword)
#             OPTIONAL MATCH (c)-[r]->(t)
#             RETURN c.name AS source, type(r) AS relation, t.name AS target LIMIT 10
#         """
        
#         with self.driver.session() as session:
#             result = session.run(cypher_query, keyword=target_entity)
#             records = [f"{rec['source']} --[{rec['relation']}]--> {rec['target']}" for rec in result]
            
#         if not records:
#             return []
            
#         return records
    
from src.core.interfaces import IRetriever
from neo4j import GraphDatabase
from gliner import GLiNER
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class Neo4jGraphRetriever(IRetriever):
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Load multi-language GLiNER model (requires ~1GB RAM)
        # First run will take some time to download the model from HuggingFace
        logger.info("Initializing GLiNER extractor...")
        self.ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        
        # Predefine the entity types we want to find
        self.labels = ["Champion", "Character", "Region", "Weapon", "Location"]

    def retrieve(self, query: str) -> list[str]:
        # 1. Extract Entity (NER) using GLiNER
        entities = self.ner_model.predict_entities(query, self.labels)
        
        # Default target entity is the first word of the query (fallback)
        target_entity = query.split()[0]
        
        # If GLiNER finds entities, take the one with the highest confidence score
        if entities:
            # Sort found entities by descending score
            best_entity = max(entities, key=lambda x: x['score'])
            target_entity = best_entity['text']
            logger.info(f"🕸️ [Graph] GLiNER detected Entity: '{target_entity}' (Label: {best_entity['label']}, Score: {best_entity['score']:.2f})")
        else:
            logger.info(f"🕸️ [Graph] GLiNER did not find any Entity, falling back to: '{target_entity}'")

        # 2. Query Neo4j with the extracted entity
        cypher_query = """
            MATCH (c)
            WHERE toLower(c.name) CONTAINS toLower($keyword)
            OPTIONAL MATCH (c)-[r]->(t)
            RETURN c.name AS source, type(r) AS relation, t.name AS target LIMIT 15
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, keyword=target_entity)
            records = [
                f"{rec['source']} --[{rec['relation']}]--> {rec['target']}" 
                for rec in result if rec['relation'] is not None
            ]
            
        return records if records else []