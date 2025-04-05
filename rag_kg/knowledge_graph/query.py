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
WHERE startNode.name IN $entitiesName
       AND startNode.type IN $entitiesType
UNWIND relationships(path) AS r
RETURN startNode AS entity1, connectedNode AS entity2, r
"""

SINGLE_HOP_RELATION_EXTRACTION_QUERY = """
MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
WHERE e1.type = $entity1Type
       AND e2.name IN $entity2Name 
       AND e2.type IN $entity2Type
       AND r.relation = $relationType
RETURN e1.name AS Entity, e1.type AS Type
"""