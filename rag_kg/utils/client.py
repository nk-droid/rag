from typing import List, Union

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()

class ClientFactory:
    @staticmethod
    def get_chat_client(
        client: str = "google",
        model: str = "gemini-2.0-flash",
        temperature: float = 0
    ) -> Union[ChatGroq, HuggingFaceEndpoint, ChatOpenAI, ChatGoogleGenerativeAI]:
        client_map = {
            "groq": ChatGroq,
            "huggingface": HuggingFaceEndpoint,
            "openai": ChatOpenAI,
            "google": ChatGoogleGenerativeAI
        }
        
        try:
            ClientClass = client_map[client]
            if client == "huggingface":
                return ClientClass(repo_id=model)
            return ClientClass(model=model, temperature=temperature)
        except KeyError:
            raise ValueError(f"Unknown client: {client}")

def get_chat_chain(
    template: str,
    input_variables: List[str],
    client: str = "google",
    model: str = "gemini-2.0-flash",
    temperature: float = 0,
    parser: PydanticOutputParser = None,
):
    client = ClientFactory.get_chat_client(client, model, temperature)
    if not parser:
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables,
        )
        return prompt | client
    
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt | client | parser