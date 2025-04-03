from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your music dataset
df = pd.read_csv("music_suggestions.csv")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")


db_location = "./music_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []


for i, row in df.iterrows():

    document = Document(
        page_content=row["Song"] + " " + row["Artist"] + " " +
        row["Review"],
        metadata={
            "rating": row["Rating"],
            "genre": row["Genre"],
            "album": row["Album"],
            "release_date": row["Release Date"]
        },
        id=str(i)
    )

    ids.append(str(i))
    documents.append(document)


vector_store = Chroma(
    collection_name="music_suggestions",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
