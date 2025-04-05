from langchain.output_parsers import EnumOutputParser, PydanticOutputParser
from rag_kg.utils.client import get_chat_chain
from rag_kg.utils.schema import ComplexityTypes, Entities, Queries, SubQueries

# TODO: Restructure

def analyze_query_complexity(query):
	template = """
You are a query analyzer system. Your task is to analyze the given queries and suggest
its complexity.

These are the complexity types:
- Single Hop: Queries that can be answered only by checking all the nodes connected to the main entity (node).
- Multiple Hop: Queries that require multiple hops to get sequential answers to for the final answer.

Here is the query separated in backticks (```):
```
{query}
```

Please provide the output in the following format:
{format_instructions}
"""

	parser = EnumOutputParser(enum=ComplexityTypes)
	chain = get_chat_chain(
		template=template,
		parser=parser,
    	input_variables=["query"]
	)
	return chain.invoke({
		"query":query
	})

def get_sequential_queries_with_dependency(query):
	template = """
You are a query generator system. Your task is to break the query into multiple queries
that can be executed in sequence to get the final answer. Use <variable_name> to store variable names that will be used in the next query.

Here is the query separated in backticks (```):
```
{query}
```

Please provide the output in the following format:
{format_instructions}
"""

	chain = get_chat_chain(
		template=template,
		parser=PydanticOutputParser(pydantic_object=SubQueries),
		input_variables=["query"]
	)

	return chain.invoke({
		"query":query
	})

def get_independent_subqueries(query):
	template = """
You are a query generator system. Your task is to analyze the given query and divide it
into multiple independent queries. Use <variable_name> to store variable names that will
be used in the next query.

Here is the query separated in backticks (```):
```
{query}
```

Please provide the output in the following format:
{format_instructions}
"""

	chain = get_chat_chain(
		template=template,
		parser=PydanticOutputParser(pydantic_object=SubQueries),
		input_variables=["query"]
	)

	return chain.invoke({
		"query":query
	}).model_dump()["subqueries"]

class EntityExtractor():
	def __init__(self, client: str = "google", model: str = "gemini-2.0-flash"):
		self.template = """
You are an NER model with the capability to identify different entities in a sentence. Given the following sentence, identify all the entities in it.

Here is the sentence separated in backticks (```):
```
{sentence}
```

Please provide the output in the following format:
{format_instructions}
"""
		self.parser=PydanticOutputParser(pydantic_object=Entities)
		self.input_variables = ["sentence"]

	def extract(self, sentence: str):
		chain = get_chat_chain(
			template=self.template,
			parser=self.parser,
			input_variables=self.input_variables
		)
		entities = chain.invoke({
			"sentence": sentence
		}).model_dump()["entities"]
		return entities
	
def create_similar_queries(query: str, n_queries: int, closest_entities: list):
    template = """
You are a similar query generator system. Your task is to use the entities and their relationships in the closest
entities list to generate {n} similar queries to the given query.

Here is the query separated in backticks (```):
```
{query}
```

Here are the closest entities related to the entities in the query separated in backticks (```):
```
{closest_entities}
```

Please provide the output in the following format:
{format_instructions}
"""

    chain = get_chat_chain(
        template=template,
        parser=PydanticOutputParser(pydantic_object=Queries),
        input_variables=["n", "query", "closest_entities"]
    )
    queries = chain.invoke({
        "n": n_queries,
        "query": query,
        "closest_entities": closest_entities
    }).model_dump()['queries']
	
    return queries