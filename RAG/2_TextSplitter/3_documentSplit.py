
from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

text = """
class Car:
    def __init__(self, brand):
        self.brand = brand

    def show(self):
        print("Brand:", self.brand)


car1 = Car("Toyota")

car1.show()"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap=0
)

res = splitter.split_text(text)
print(len(res))
print(res)