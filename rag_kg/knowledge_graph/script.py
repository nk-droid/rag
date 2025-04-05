import json
import os
from neo4j import GraphDatabase
from rag_kg.knowledge_graph.query import *

DATA_DIR = "./rag_kg/data"

driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def load_relations():
    with open(os.path.join(DATA_DIR, "processed", "wikisent2_filtered.json"), 'r', encoding='utf-8') as json_file:
        relations = json.load(json_file)

    return relations

def populate_database():
    relations = load_relations()
    with driver.session() as session:
        session.run(query=GRAPHDB_POPULATION_QUERY, relations=relations)

def get_all_entities_relations():
    with driver.session() as session:
        return session.run(query=ALL_ENTITIES_RELATION_EXTRACTION_QUERY)

def get_entity_relation(entities):
    with driver.session() as session:
        result = session.run(
            ENTITY_RELATION_EXTRACTION_QUERY,
            entitiesName=[entity['name'] for entity in entities],
            entitiesType=[entity['type'] for entity in entities] 
        )

        return list(result)

def get_single_hop_entities(entity1_type, entity2, relation_type):
    with driver.session() as session:
        single_hop_entities = session.run(
            SINGLE_HOP_RELATION_EXTRACTION_QUERY,
            entity1Type=entity1_type,
            entity2Name=[entity['Entity'] for entity in entity2],
            entity2Type=[entity['Type'] for entity in entity2],
            relationType=relation_type
        )

        return [dict(entity) for entity in list(single_hop_entities)]