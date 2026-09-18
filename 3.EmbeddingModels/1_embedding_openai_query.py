from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbedding(model='text-embedding-3-large',dimension = 32)

text ='capital of India is New Delhi'

res = embedding.embed_query(text)
print(str(res))
