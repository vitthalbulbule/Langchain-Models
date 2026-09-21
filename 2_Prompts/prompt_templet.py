from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    
)

st.header('Summarization tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(template= """
You are an expert research paper explainer.

Explain the following research paper:

Research Paper: {paper}

Explanation Style: {style}

Explanation Length: {length}

Follow these instructions:

1. Start with a brief introduction to the paper.
2. Explain the main problem the paper tries to solve.
3. Explain the key idea or proposed approach.
4. Explain the important concepts introduced in the paper.
5. Explain the architecture or methodology, if applicable.
6. Explain the results and significance of the paper.
7. Give a simple real-world analogy or example where appropriate.
8. Keep the explanation consistent with the selected explanation style and length.
9. Avoid unnecessary complexity unless the selected style is Technical or Mathematical.

Provide the explanation in a clear and well-structured format.
""",
input_variables=["paper", "style", "length"] )


prompt = template.invoke({
"paper": paper_input, "style": style_input, "length": length_input
})


if st.button('Generate Summary'):
    result = llm.invoke(prompt)
    st.write(result.content)


