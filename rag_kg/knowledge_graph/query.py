GRAPHDB_POPULATION_QUERY = """
UNWIND $relations AS relation
MERGE (entity1:Entity {name: relation.entity1.name, type: relation.entity1.type})
MERGE (entity2:Entity {name: relation.entity2.name, type: relation.entity2.type})
MERGE (entity1)-[:RELATED_TO {relation: relation.relation}]->(entity2)
"""

ALL_ENTITIES_RELATION_EXTRACTION_QUERY = """
MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
RETURN e1.name AS Entity1, e1.type AS Type1, 
       e2.name AS Entity2, e2.type AS Type2,
       r.relation AS Relation
"""

ENTITY_RELATION_EXTRACTION_QUERY = """
MATCH path = (startNode)-[*1..3]-(connectedNode)
WHERE startNode.name IN $entitiesName AND startNode.type IN $entitiesType
UNWIND relationships(path) AS rel
RETURN startNode AS entity1, connectedNode AS entity2, rel
"""