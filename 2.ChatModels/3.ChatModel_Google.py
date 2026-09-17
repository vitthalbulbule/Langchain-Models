from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="chat-bison-001")

res = model.invoke("Write a poem about a lonely computer.")
print(res.content)
