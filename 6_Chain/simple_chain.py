from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace


load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='Generate new ersion 5 intrested fact about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

res = chain.invoke({'topic':'Virat Kohali'})

print(res)

chain.get_graph().print_ascii(  )