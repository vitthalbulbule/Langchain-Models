from langchain_community.document_loaders import TextLoader

loader = TextLoader('Cricket_Basic_Information.txt')
document = loader.load()

print(document)