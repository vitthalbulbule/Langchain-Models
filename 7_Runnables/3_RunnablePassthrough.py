from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough
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
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Give the 5 line explanation about the joke {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'expla':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'Cricket'}))