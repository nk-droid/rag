"""
PIPELINE WORKFLOW (Baseline)
1. Create Knowledge Graph
2. Check complexity of user's query
    a) If simple query (single hop), provide results.
    b) If complex query (multi hops),
        i) Break the query into multiple simple queries
        ii) Pull one query at a time in a topologically sorted manner
        iii) Check the queries on which this query is dependent
        iv) Pass context from the dependent queries to get results

OPTIMISATION
1. Process independent queries in parallel.
2. Add caching
"""