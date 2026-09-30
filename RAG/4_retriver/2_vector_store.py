from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    Document(
        page_content="""Virat Kohli is an Indian international cricketer , he is batter""",
        metadata={"player": "Virat Kohli", "country": "India"}
    ),
    Document(page_content="""langcahin is an framework""",
             metadata={''}
             )]


