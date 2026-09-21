from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import streamlit as st
load_dotenv()


st.header('Hugginh Face Chat Model API')

user_input = st.text_input('Enter your query here : ')



llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    
)

model = ChatHuggingFace(llm=llm)


if st.button('Submit'):
    res = model.invoke(user_input)
    st.write(res.content)