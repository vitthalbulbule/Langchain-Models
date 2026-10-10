from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

res = model.invoke('Who is the better cricketer in the world ?')

print(res.content)


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# import simple as st
# load_dotenv()


# st.header('Hugginh Face Chat Model API')

# user_input = st.text_input('Enter your query here : ')



# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     task="text-generation",
    
# )

# model = ChatHuggingFace(llm=llm)


# if st.button('Submit'):
#     res = model.invoke(user_input)
#     st.write(res.content)