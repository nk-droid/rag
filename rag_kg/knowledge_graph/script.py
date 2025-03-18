import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
session = driver.session()

def populate_db():
    pass

def get_all_entities_relations():
    pass

def get_entity_relation():
    pass