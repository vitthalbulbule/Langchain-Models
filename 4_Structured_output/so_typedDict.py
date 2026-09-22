from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='conversational'
)

model = ChatHuggingFace(llm=llm)

# Creating Schema
class review(TypedDict):
    summary : str
    sentiment : str

structured_model = model.with_structured_output(review)


res = structured_model.invoke("""The Samsung Galaxy S25 is a premium smartphone with a bright AMOLED display and powerful performance. Its camera system produces detailed photos, and the phone handles gaming and multitasking smoothly. The battery easily lasts a full day with normal usage. The software experience is clean and responsive, with several useful AI features.

However, the phone is relatively expensive, and the battery capacity could be larger. Overall, it is a well-balanced smartphone for users who want strong performance, a good camera, and a premium design.""")

print(res)