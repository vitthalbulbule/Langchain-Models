from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Give the answer of the {question} using this {text}',
    input_variables=['question','text']
)

url = 'https://www.flipkart.com/a/p/itm1692bd8b2fe84?pid=COMHPDSBSZF5ZF7B&PARAM=3738&pageUID=1790400065937'
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'Give the product specification','text':docs[0].page_content}))