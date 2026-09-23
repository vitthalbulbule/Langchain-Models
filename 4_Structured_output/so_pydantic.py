from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate

load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):

    name : str=Field(description='Give the name of the mobile')
    rating : float = Field(description="Rating of the mobile phone out of 10")
    pros : list[str] = Field(description='give the pros in simple points or advantages of the phone' )
    cons: list[str] = Field(description='give the cons in simple points or disadvantages of the phone')

template = PromptTemplate(
    template='give the review of the {phone} , in simple format',
    input_variables=['phone']
)
# prompt = template.invoke({'phone':'oppo'})

structured_outputt = model.with_structured_output(Review,method='json_schema')

# res = structured_outputt.invoke(prompt)


chain = template | structured_outputt

res = chain.invoke({'phone':'samsung altra'})

print(res)


# from huggingface_hub import InferenceClient
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os
# import json

# load_dotenv()


# class Review(BaseModel):
#     name: str
#     rating: float
#     pros: list[str]
#     cons: list[str]


# client = InferenceClient(
#     provider="auto",
#     api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )


# response_format = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "Review",
#         "schema": Review.model_json_schema(),
#         "strict": True
#     }
# }


# messages = [
#     {
#         "role": "system",
#         "content": "Give a structured review of the mobile phone."
#     },
#     {
#         "role": "user",
#         "content": "Give me a review of the Samsung Galaxy S25."
#     }
# ]


# response = client.chat.completions.create(
#     model="Qwen/Qwen3-32B",
#     messages=messages,
#     response_format=response_format
# )


# result = json.loads(response.choices[0].message.content)

# print(result)