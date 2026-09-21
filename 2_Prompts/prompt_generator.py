from langchain_core.prompts import PromptTemplate

from langchain_core.load import dumps


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


# Convert LangChain object to JSON
json_data = dumps(template)

# Save JSON to file
with open("template.json", "w", encoding="utf-8") as f:
    f.write(json_data)

print("Template saved successfully!")