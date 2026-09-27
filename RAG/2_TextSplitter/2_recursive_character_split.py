# more useful for current RAG based application

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap=0
)


text = """ 
I am Vitthal 
I am 20 years old

I live in pune
How are you
"""
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)