from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    
)

model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template='Give the detail summary on the {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='give the 5 line summmary on the {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({'topic':'langchain'})

print(res.content)