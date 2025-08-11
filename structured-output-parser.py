from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm= llm)

schema= [
    ResponseSchema(name= 'fact1', description= ' fact 1 about the topic'),
    ResponseSchema(name= 'fact2', description= ' fact 2 about the topic'),
    ResponseSchema(name= 'fact3', description= ' fact 3 about the topic'),
]

parser= StructuredOutputParser.from_response_schemas(schema)

temp = PromptTemplate(
    template= 'give 3 facts about the {topic} \n {format_instruction}',
    input_variables= ['topic'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = temp | model | parser 

res= chain.invoke({'topic': 'black hole'})

print(res)
