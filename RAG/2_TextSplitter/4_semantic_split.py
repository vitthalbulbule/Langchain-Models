from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text = """Python is a programming language. It is easy to learn and widely used.
Machine learning is a branch of AI. It allows computers to learn from data.
Deep learning uses neural networks to learn complex patterns."""

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)

print(chunks)