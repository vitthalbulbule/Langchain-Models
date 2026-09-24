from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B',
    task="conversational"
)

model =ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Give a short summary about {topic} in 5 points.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Give five interesting facts about the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 |model |parser | template2 | model | parser

result = chain.invoke({'topic':'Unemployment in India'})

print(result)