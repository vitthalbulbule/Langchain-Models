from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Give the five best diciplines on the man {man}',
    input_variables=['man']
)

chain = RunnableSequence(prompt,model,parser)

res = chain.invoke({'man':'Virat Kohali'})

print(res)