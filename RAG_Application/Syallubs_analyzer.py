from langchain_huggingface import HuggingFaceEmbeddings , ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

# Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_chroma import Chroma

# For using MMR does not require to import statement

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 1. Load PDF

loader = PyPDFLoader('RAG_Application\TECESyllabus-Draft 14-5-2026 (1).pdf')
documentss = loader.load()

# 2. Text Splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap = 20
)

split_text = splitter.split_documents(documentss)

# 3 & 4 Embedding + Vector Store

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma_db',
    collection_name='Syallubs_storage'
)

vector_store.add_documents(split_text)

# Retrival

retrivar = vector_store.as_retriever(
    search_type='mmr',
    search_lwargs={'k':2}
)

query = 'Give me the exact unit wise syallubs for the Artificial Intelligence subject ?'
docs = retrivar.invoke(query)

context = "\n\n".join(
    doc.page_content for doc in docs
)

from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template= 'Answer the question using the following context. Context {context} Question {question}',
    input_variables=['context','query']
)

res = model.invoke(template)

for i,r in res:
    print("*** {i+1} ***")
    print(r.page_content)