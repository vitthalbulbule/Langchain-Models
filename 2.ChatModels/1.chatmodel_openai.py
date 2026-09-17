from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

res = model.invoke('Who is the best cricket player in the world?')
print(res)