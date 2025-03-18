import json
import os
from neo4j import GraphDatabase
from rag_kg.knowledge_graph.query import *

DATA_DIR = "./rag_kg/data"

driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
session = driver.session()

# Function to clear the database
def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def load_relations():
    with open(os.path.join(DATA_DIR, "processed", "wikisent2_filtered.json"), 'r', encoding='utf-8') as json_file:
        relations = json.load(json_file)

    return relations

def populate_database():
    relations = load_relations()
    return session.run(query=GRAPHDB_POPULATION_QUERY, relations=relations)

def get_all_entities_relations():
    return session.run(query=ALL_ENTITIES_RELATION_EXTRACTION_QUERY)

def get_entity_relation(entities):
    return session.run(
        ENTITY_RELATION_EXTRACTION_QUERY,
        entitiesName=[entity['name'] for entity in entities],
        entitiesType=[entity['type'] for entity in entities] 
    )