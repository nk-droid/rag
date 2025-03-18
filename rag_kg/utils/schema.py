from typing import List
from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict, deque

class EntityTypes(str, Enum):
	CITY: str = "city"
	COUNTRY: str = "country"
	PERSON: str = "person"
	ORG: str = "organization"
	DATE: str = "date"
	EVENT: str = "event"
	BOOK: str = "book"
     
class RelationTypes(str, Enum):
	LOCATED_IN = "located_in"
	BORN_IN = "born_in"
	FOUNDED_IN = "founded_in"
	HAPPENED_ON = "happened_on"
	HELD_IN = "held_in"
	HELD_ON = "held_on"
	ASSOCIATED_WITH = "associated_with"
	TRAVELED_TO = "traveled_to"
	ORGANIZED_BY = "organized_by"
	DATED_ON = "dated_on"
	STUDIED_AT = "studied_at"
	VISITED = "visited"
	CREATED_BY = "created_by"
	HOSTED_BY = "hosted_by"
	WRITTEN_BY = "written_by"
	PUBLISHED_BY = "published_by"
	PUBLISHED_ON = "published_on"
	KNOWN_AS = "known_as"

class Entity(BaseModel):
	name: str = Field(..., title="Name of the entity")
	type: EntityTypes = Field(..., title="Type of the entity")

	class Config:  
		use_enum_values = True
		
class Entities(BaseModel):
	entities: List[Entity]

class Relation(BaseModel):
	entity1: Entity
	entity2: Entity
	relation: RelationTypes = Field(..., title="How is entity2 related to entity1 (in at max 2 words)?")
	
	class Config:  
		use_enum_values = True

class Relations(BaseModel):
	relations: List[Relation]

class ComplexityTypes(str, Enum):
	SINGLE_HOP: str = "single_hop"
	MULTIPLE_HOP: str = "multiple_hop"
	
class Queries(BaseModel):
	queries: List[str] = Field(..., title="List of queries to be executed")

class SubQuery(BaseModel):
	query: str = Field(..., title="Query to be executed")
	dependencies: List[int] = Field(..., title="A list of indices of the queries that need to be executed before this query")

class SubQueries(BaseModel):
	subqueries: List[SubQuery]

	def yield_subquery_idx_to_execute(self): # TODO: Check this function
		"""Yields subqueries in a topologically sorted manner"""
		
		# Build adjacency list and in-degree array
		adjacency = defaultdict(list)
		in_degree = [0] * len(self.subqueries)

		# Fill adjacency and in_degree
		for i, subquery in enumerate(self.subqueries):
			for dep_idx in subquery.dependencies:
				adjacency[dep_idx].append(i)
			in_degree[i] = len(subquery.dependencies)

		# Initialize queue with all subqueries that have zero in-degree
		queue = deque([i for i, deg in enumerate(in_degree) if deg == 0])

		# Topological sort
		while queue:
			current = queue.popleft()
			yield self.subqueries[current]  # yield current subquery

			for neighbor in adjacency[current]:
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					queue.append(neighbor)
					     
# TODO: Implement File schema
class File(BaseModel):
    filepath: str

# TODO: Implement Metadata schema 
class Metadata(BaseModel):
    name: str
    description: str
    version: str