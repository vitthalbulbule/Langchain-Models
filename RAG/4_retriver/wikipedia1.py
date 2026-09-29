# from langchain_community.retrievers import WikipediaRetriever

# retriever = WikipediaRetriever(
#     top_k_results=2
# )

# query = "Virat Kohli best ODI innings"

# res = retriever.invoke(query)

# for doc in res:
#     print(doc.page_content)
#     print(doc.metadata)
#     print("-" * 50)


import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="MyRAGProject/1.0",
    language="en"
)

page = wiki.page("Virat Kohli")

print(page.exists())
print(page.summary)