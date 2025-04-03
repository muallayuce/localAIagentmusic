from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")


template = """
You are an expert in recommending music based on user preferences and moods.

Here are some relevant songs and their details: {songs_data}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question about music (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break

    songs_data = retriever.invoke(question)
    result = chain.invoke({"songs_data": songs_data, "question": question})
    print(result)
