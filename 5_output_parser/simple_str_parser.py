from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt = template1.invoke({'topic':'langchain'})

result = model.invoke(prompt)

prompt2 = template2.invoke({'text':result.content})

res = model.invoke(prompt2)

print(res.content)

