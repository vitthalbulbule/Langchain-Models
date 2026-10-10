from langchain_huggingface import HuggingFaceEmbeddings , ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

# Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# For using MMR does not require to import statement

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="conversational"
)

# llm = HuggingFaceEndpoint(
#     repo_id='XiaomiMiMo/MiMo-V2.6-Pro-RL',
#     task="conversational"
# )



model = ChatHuggingFace(llm=llm)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 1. Load PDF

loader = PyPDFLoader('RAG_Application/TECESyllabus-Draft 14-5-2026 (1).pdf')
documentss = loader.load()

# 2. Text Splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 100
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
    
)




# docs = retrivar.invoke(query)

# context = "\n\n".join(
#     doc.page_content for doc in docs
# )

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retrivar | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})



template = PromptTemplate(
    template= """Answer the Following question using the provided context. 
    if the context is not sufficient , simple saind ,'i dont know..! Please provide correct Info'
    {context} 
    Question : {question}""",
    input_variables=['context','question']
)

parser = StrOutputParser()

final_chain = parallel_chain | template | model | parser


question = 'List the all Computer Network Laboratory praticles ?'
res = final_chain.invoke(question)

print(res)