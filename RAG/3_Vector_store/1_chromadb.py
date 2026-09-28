from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document

documents = [
    Document(
        page_content="""Virat Kohli is an Indian international cricketer known for his batting ability,
        consistency, and aggressive style of play. He has represented India in all three formats of cricket.
        Kohli has played many important innings for India and has been one of the leading run-scorers
        in international cricket. He has also captained the Indian cricket team.""",
        metadata={"player": "Virat Kohli", "country": "India"}
    ),

    Document(
        page_content="""Rohit Sharma is an Indian cricketer known for his elegant batting and ability
        to score big hundreds. He has represented India in Test, ODI, and T20 cricket. Rohit is also
        known for his leadership and has captained India in international cricket. He has scored
        multiple double centuries in ODI cricket.""",
        metadata={"player": "Rohit Sharma", "country": "India"}
),
        Document(
        page_content="""Sachin Tendulkar is a former Indian cricketer and one of the most famous
        batsmen in cricket history. He represented India for more than two decades. Tendulkar
        scored 100 international centuries and became the first player to score a double century
        in men's ODI cricket. He was part of India's 2011 World Cup winning team.""",
        metadata={"player": "Sachin Tendulkar", "country": "India"}
    )]
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma-db',
    collection_name='sample'
)

# 4. Convert documents into embeddings and store them
# vector_store.add_documents(documents)

# print("Documents added successfully!")
# print("Number of documents:", vector_store._collection.count())

# print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

# que = vector_store.similarity_search_with_score(
#     query='Who is the aggressive Batsman ?',
#     k=2
# )

# print(que)

result = vector_store.similarity_search_with_score(
    "aggressive style of play",
    k=3
)

for doc, score in result:
    print(doc.metadata["player"], score)