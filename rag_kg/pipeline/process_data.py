import re
import os
import json

import time
from typing import List

from rag_kg.utils.schema import Relations
from rag_kg.utils.client import get_chat_chain
from langchain.output_parsers import PydanticOutputParser

DATA_DIR = "./rag_kg/data"

def filter_data(cities: List[str]) -> List[str]:
    city_pattern = r'\b(?:' + '|'.join(map(re.escape, cities)) + r')\b'
    
    with open(os.path.join(DATA_DIR, "raw", "wikisent2.txt"), 'r', encoding='utf-8') as file:
        filtered_lines = [line.strip() for line in file if re.search(city_pattern, line, re.IGNORECASE)]
    
    return filtered_lines

def dump_data(results):
    with open(os.path.join(DATA_DIR, "processed", "wikisent2_filtered.json"), 'w', encoding='utf-8') as json_file:
	    json.dump(results, json_file, indent=4)

def create_entity_relation_data(sentences: List[str]):
    template = """
You are an NER model with capability to identify relationships between entities in a sentence. Given the following sentence,
identify all the entity pairs with relationship between them from the following list:
- located_in
- born_in
- founded_in
- happened_on
- held_in
- held_on
- associated_with
- traveled_to
- organized_by
- dated_on
- studied_at
- launched_at
- visited
- created_by
- hosted_by
- written_in
- written_by
- directed_by
- produced_by
- published_by
- published_on
- known_as

Here is the sentence separated in backticks (```):
```
{sentence}
```

Please provide the output in the following format:
{format_instructions}
"""

    chain = get_chat_chain(
        template=template,
        parser=PydanticOutputParser(pydantic_object=Relations),
        input_variables=["sentence"]
    )
    
    all_relations = []
    try:
        for idx, sentence in enumerate(sentences):
            relations = chain.invoke({
                "sentence": sentence
            }).model_dump()["relations"]
            all_relations.extend(relations)
            time.sleep(3) # FIXME: Using for gemini rpm limit

        return all_relations
    except Exception as e:
        print(f"Error processing sentence: {sentence}")
        print(f"Exception: {e}")
        return []
    finally:
        dump_data(all_relations)
