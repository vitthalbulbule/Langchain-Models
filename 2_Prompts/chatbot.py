from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='conversational'
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='you are helpful assistant like teacher')
]        # For storing previous chats

while True:

    user_input=input('You :')
    chat_history.append(HumanMessage(content=user_input) )   # Adding Each input to the history
    if user_input=='exit':
        break

    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))  # Also add result in the history as well
    print('AI :',result.content)

print(chat_history)

