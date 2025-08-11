from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm= llm)

class person(BaseModel):
    name: str = Field(description= ' name of the person')
    age: int = Field(gt= 18, description= 'age of the person')
    city: str = Field(description= 'name of gthe city the person belongs to')

parser = PydanticOutputParser(pydantic_object= person)

temp = PromptTemplate(
    template= 'generate the name, age and city of the {place} person \n {format_instruction}',
    input_variables= ['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = temp | model | parser

res = chain.invoke({'place': 'srilankan'})

print(res)