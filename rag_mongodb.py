import os
import time
from dotenv import load_dotenv
from typing import List

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

# ENVIRONMENT SETUP
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MongoURI")

if not OPENAI_API_KEY or not MONGO_URI:
    raise EnvironmentError("Missing required environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGO_URI)

DB_NAME = "RAG"
COLLECTION_NAME = "documents"
INDEX_NAME = "vector_index"

collection = mongo_client[DB_NAME][COLLECTION_NAME]

# EMBEDDINGS
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # MUST match model

def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# INGESTION
def ingest_pdf(pdf_url: str):
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    docs = []
    for doc in chunks:
        docs.append({
            "text": doc.page_content,
            "embedding": get_embedding(doc.page_content),
            "metadata": {
                "page": doc.metadata.get("page"),
                "source": pdf_url
            }
        })

    if docs:
        collection.insert_many(docs)
        print(f"Inserted {len(docs)} documents")

# VECTOR INDEX
def create_vector_index():
    existing = list(collection.list_search_indexes(INDEX_NAME))
    if existing:
        print("Vector index already exists")
        return

    index_model = SearchIndexModel(
        name=INDEX_NAME,
        type="vectorSearch",
        definition={
            "fields": [{
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIM,
                "similarity": "cosine"
            }]
        }
    )

    collection.create_search_index(index_model)
    print("Creating vector index...")

    while True:
        indexes = list(collection.list_search_indexes(INDEX_NAME))
        if indexes and indexes[0].get("queryable"):
            break
        time.sleep(5)

    print("Vector index is ready")

# RETRIEVAL
def vector_search(query: str, k: int = 5) -> List[str]:
    query_embedding = get_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1
            }
        }
    ]

    results = collection.aggregate(pipeline)
    return [doc["text"] for doc in results]


# RAG PROMPT + LLM
RAG_PROMPT = PromptTemplate(
    template="""
You are a precise AI assistant.

Answer the question strictly using the provided context.
If the answer is not found in the context, say:
"I am unable to find the requested information in the provided documents."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)

def answer_query(query: str) -> str:
    docs = vector_search(query)

    if not docs:
        return "I am unable to find the requested information in the provided documents."

    context = "\n\n".join(docs)

    chain = RAG_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content

# MAIN
if __name__ == "__main__":
    PDF_URL = "https://investors.mongodb.com/node/12236/pdf"

    ingest_pdf(PDF_URL)
    create_vector_index()

    query = "What are MongoDB's latest AI announcements?"
    answer = answer_query(query)

    print("\nFinal Answer:\n", answer)
