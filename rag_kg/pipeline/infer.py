from rag_kg.config.logger import get_logger
from rag_kg.utils.schema import ComplexityTypes
from rag_kg.llm_utils.query_processing import analyze_query_complexity, get_independent_subqueries, get_sequential_queries_with_dependency
from rag_kg.knowledge_graph.script import get_single_hop_entities

from rag_kg.utils.client import get_chat_chain

logger = get_logger(__name__)

def query_planning(query):
    """
    Plans the sequence in which the query should be broken and processed.
    """
    complexity = analyze_query_complexity(query)
    if complexity == ComplexityTypes.SINGLE_HOP:
        return get_independent_subqueries(query)
    
    elif complexity == ComplexityTypes.MULTIPLE_HOP:
        sequential_queries = get_sequential_queries_with_dependency(query)
        inpendent_queries = []
        for subquery in sequential_queries.yield_subquery_to_execute():
            inpendent_queries.append(subquery)

        return inpendent_queries
    
    else:
        raise ValueError(f"Unknown complexity type: {complexity}")

def resolve_dependency(query, extracted_entities):
	dependent_entities = []
	if query['dependencies']:
		for dep in query['dependencies']:
			dependent_entities.extend(extracted_entities[dep])

		return dependent_entities
	else:
		return [{"Type": query['entity2']['type'], "Entity": query['entity2']['name']}]
     
def get_context_data(query):
    logger.info(f"Processing Query: {query}")
    independent_queries = query_planning(query)

    logger.info(f"Independent Queries: {independent_queries}")
    extracted_entities = {}
    for queries in independent_queries:
        # TODO: Parallelize this part
        for query in queries:
            entity2 = resolve_dependency(query, extracted_entities)
            entities = get_single_hop_entities(query['entity1_type'], entity2, query['relation'])
            extracted_entities[query['index']] = entities

    extracted_entities_from_last_queries = {}
    for query in independent_queries[-1]:
        extracted_entities_from_last_queries[query['index']] = extracted_entities[query['index']]

    return independent_queries[-1], extracted_entities_from_last_queries

def get_query_response(query, last_independent_queries, extracted_entities):
    template = """
You are a component of a RAG system. The previous components of the system have broken the query into multiple queries and
extracted the entities for the final independent queries from the knowledge graph. Your task is to generate a response
based on the original query, the last independent queries and their corresponding extracted entities.

Here is the query separated in backticks (```):
```
{query}
```

Here are the last independent queries separated in backticks (```):
```
{last_independent_queries}
```

Here are the extracted entities separated in backticks (```):
```
{extracted_entities}
```
"""

    chain = get_chat_chain(
        template=template,
        input_variables=["query", "last_independent_queries", "extracted_entities"],
    )
    return chain.invoke({
        "query":query,
        "last_independent_queries": last_independent_queries,
        "extracted_entities": extracted_entities
    }).content


def get_result(query):
    last_independent_queries, extracted_entities = get_context_data(query)
    return get_query_response(query, last_independent_queries, extracted_entities)
