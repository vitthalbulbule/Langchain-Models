# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from langchain_core.runnables import RunnableParallel , RunnableBranch ,RunnableLambda
# from pydantic import BaseModel ,Field
# from langchain_core.output_parsers import PydanticOutputParser
# from typing import Literal


# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='Qwen/Qwen3-32B',
#     task="conversational"
# )

# model = ChatHuggingFace(llm=llm)

# class Feedback(BaseModel):

#     sentiment : Literal['positive','negative'] = Field(description='Give the sentement of the feedback')

# parser2=PydanticOutputParser(pydantic_object=Feedback)



# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the  give customer feedback text as positive or negative  {feedback} \n {format_instructions}',
#     input_variables=['feedback'],
#     partial_variables={'format_instructions':parser2.get_format_instructions()}
# )
# parser = StrOutputParser()

# classifier_chain = prompt1 | model | parser2






# prompt2 = PromptTemplate(
#     template='Give the positive feedback to the user {feedback}',
#     input_variables=['feedback']
# )

# prompt3 = PromptTemplate(
#     template='Give the Negative feedback to the user {feedback}',
#     input_variables=['feedback']
# )

# branch = RunnableBranch(
#     (lambda x:x.sentiment=='positive', prompt2 | model | parser),
#     (lambda x:x.sentiment=='negative' , prompt3 | model | parser),
#     RunnableLambda(lambda x:'Could not find the sentiment')

# )


# final_chain = classifier_chain | branch

# res = final_chain.invoke({'feedback':'This is a wonderful phone'})

# print(res)


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template="""Classify the sentiment of the given customer feedback
    as positive or negative.

    Feedback: {feedback}

    {format_instructions}""",
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': parser2.get_format_instructions()
    }
)

parser = StrOutputParser()


classifier_chain = prompt1 | model | parser2


prompt2 = PromptTemplate(
    template='Give a positive response to the user about this feedback: {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Give a negative response to the user about this feedback: {feedback}',
    input_variables=['feedback']
)


branch = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not find the sentiment')
)


final_chain = classifier_chain | branch

res = final_chain.invoke({
    'feedback': 'This is a wonderful phone'
})

print(res)