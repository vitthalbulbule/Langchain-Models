from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('1_RAG_DocumentLoader\Syanopsis CCTV Survailance.pdf')

docs = loader.load()

print(docs)