# api_key = 'pcsk_7GRqgy_UVMs33VV7JiNnuhn1f2DmSS78g3wvHUu9azfjMcYaqL7pB4t2tx7HnAxZJkjMvd'

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
