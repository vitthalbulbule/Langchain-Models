from langchain_core.runnables import RunnableSequence ,RunnablePassthrough , RunnableParallel ,RunnableLambda , RunnableBranch
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


prompt1 = PromptTemplate(
    template='generate a detail summary about  {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the text {text}',
    input_variables=['text']
)

sum_chain = RunnableSequence(prompt1 , model , parser)

cond_chain = RunnableBranch(
    (lambda x:len(x.split())>100 ,RunnableSequence(prompt1 | model | parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(sum_chain,cond_chain)

print(final_chain.invoke({'topic':'India Job Market'}))