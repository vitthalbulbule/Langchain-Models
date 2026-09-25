from langchain_core.runnables import RunnableSequence , RunnableParallel
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
    template='Generate the 2 Advantages about the {man}',
    input_variables=['man']
)

prompt2 = PromptTemplate(
    template='Give the 2 Disadvantages about the {man}',
    input_variables=['man']
)

p_chain = RunnableParallel({
    'advan':prompt1 | model | parser,
    'disadv': prompt2 | model | parser,
})

res = p_chain . invoke({'man':'Virat Kohali'})
print(res)
