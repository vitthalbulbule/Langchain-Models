from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task="conversational"
)

llm2 = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B',
    task="conversational"
)
model1 =ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = 'Generate a short notes on the given \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Give the 5 question and answers on the give {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz and return a single result.\n\nNotes:\n{notes}\n\nQuiz:\n{questions}',
    input_variables=['notes', 'questions']
)

parser = StrOutputParser()


# Parallel Chain
paraller_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'questions': prompt2 | model2 | parser
})

#Sequential Chain for prompt3

seq_chain = prompt3 | model1 | parser

# final merge chain

chain = paraller_chain | seq_chain

text = """ What is a neural network?
A neural network is a machine learning model that stacks simple "neurons" in layers and learns pattern-recognizing weights and biases from data to map inputs to outputs.

Neural networks are among the most influential algorithms in modern machine learning and artificial intelligence (AI). They underpin breakthroughs in computer vision, natural language processing (NLP), speech recognition and countless real-world applications ranging from forecasting to facial recognition. While today’s deep neural networks (DNNs) power systems as complex as transformers and convolutional neural networks (CNNs), the origins of neural networks trace back to simple models such as linear regression and how the human brain digests, processes and decides on the information presented to it.

How do neural networks work?
On a high level, the inspiration for neural networks comes from the biological neurons in the human brain, which communicate through electrical signals. In 1943, Warren McCulloch and Walter Pitts proposed the first mathematical model of a neuron, showing that simple units could perform computation of a function. Later, in 1958, Frank Rosenblatt introduced the perceptron, an algorithm designed to perform pattern recognition. The perceptron is the historical ancestor of today’s networks: essentially a linear model with a constrained output. In the following section, we will dive into how neural networks borrow inspiration from the human brains to make decisions and recognize patterns.  

A neural network can be understood through a simple example: spam detection. An email is fed into the network, and features such as words or phrases like “prize,” “money,” “dear” or “win” are used as inputs. The early neurons in the network process the importance of each signal, while later layers combine this information into higher-level cues that capture context and tone. The final layer then computes a probability of whether the email is spam, and if that probability is high enough, the email is flagged. In essence, the network learns how to transform raw features into meaningful patterns and use them to make predictions.

This process is powered by two fundamental concepts: weights and biases. Weights act like dials that control how strongly each input feature influences the decision—a word like “prize” may be given more weight than a common word like “hello.” Biases are built-in values that shift the decision threshold, allowing a neuron to activate even if the inputs themselves are weak. Together, these model parameters determine how each neuron contributes to the overall computation. By adjusting these values during training, the network gradually learns to make accurate predictions—in this case, whether an email is spam or not.

  """

res = chain.invoke({'text':text})
print('Running')

print(res)