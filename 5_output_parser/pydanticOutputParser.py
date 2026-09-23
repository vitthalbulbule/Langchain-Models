from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

class People(BaseModel):

    name : str=Field(description='Give the name of the person')
    age : int = Field(gt=18,description='age of the person')
    city : str = Field(description='city of the person')

parser = PydanticOutputParser(pydantic_object=People)

template = PromptTemplate(
    template='give name ,age and city of cricketer in  {country}\n {format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser

res = chain.invoke({'country':'India'})


print(res.content)