from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = 'task-generation'

)

model = ChatHuggingFace(llm=llm)

message = [
    SystemMessage(content='You are a helpful ai assistant'),
    HumanMessage(content='Tell me something about langchain')
]

res = model.invoke(message)

message.append(AIMessage(content=res.content))

print(message)