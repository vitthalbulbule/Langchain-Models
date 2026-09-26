from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='PDF',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)     # First PDF First page
# print(docs[0].metadata)

docs = loader.lazy_load()

for d in docs:
    print(d.metadata)