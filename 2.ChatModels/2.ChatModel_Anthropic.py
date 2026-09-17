from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2")

res = model.invoke("Write a poem about a lonely computer.")

print(res.content)
