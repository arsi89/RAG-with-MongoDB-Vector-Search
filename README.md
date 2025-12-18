# RAG with MongoDB Vector Search 🔍📚

**Repository:** RAG (Retrieval-Augmented Generation) using MongoDB Vector Search

This project demonstrates a simple RAG pipeline that:

- Ingests PDF documents, splits them into chunks, and creates embeddings using OpenAI.
- Stores document chunks and their embeddings in a MongoDB collection.
- Creates a MongoDB vector search index and performs semantic retrieval.
- Uses a language model to generate precise answers constrained to retrieved context.

---

## ✅ Features

- PDF ingestion (uses `PyPDFLoader`)
- Text chunking with overlap for better retrieval quality
- Embeddings generated via OpenAI embeddings API
- Vector search using MongoDB Search Index (vectorSearch)
- Simple RAG prompt + LLM answer generation

---

## 🔧 Requirements

Make sure you have the following installed (see `requirements.txt`):

- Python 3.10+
- streamlit, langchain, OpenAI, PyPDFLoader, python-dotenv, etc.

Additional required packages for this project (if not already present):

```bash
pip install pymongo openai
```

Note: The project uses the `openai` client for embeddings and `pymongo` for MongoDB access.

---

## 🧰 Setup

1. Clone this repository and create a venv:

```bash
git clone <repo-url>
cd mongodb_rag
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install pymongo openai
```

2. Create a `.env` file in the project root with the following values:

```env
OPENAI_API_KEY=sk-...
MongoURI=mongodb+srv://<user>:<pass>@cluster0.mongodb.net
```

- **OPENAI_API_KEY**: API key for embeddings and LLM calls.
- **MongoURI**: Your MongoDB connection string. This project expects a MongoDB deployment that supports Search/Vector indexes (e.g., MongoDB Atlas with Atlas Search vector support or an appropriate server version).

> Note: The environment variable name used in `rag_mongodb.py` is `MongoURI` (case-sensitive), so use the same name.

---

## ⚙️ How it works (quick walkthrough)

- `ingest_pdf(pdf_url)`
  - Loads the PDF, splits it into chunks, computes embeddings, and stores documents in the `documents` collection.

- `create_vector_index()`
  - Creates a MongoDB Search index of type `vectorSearch` on the `embedding` field.

- `vector_search(query, k=5)`
  - Embeds the query and runs a `$vectorSearch` aggregation to retrieve the top-k most similar chunks.

- `answer_query(query)`
  - Uses the retrieved chunks as `context` and sends them to an LLM with a strict prompt that forbids hallucination.

---

## ▶️ Run the demo

The main script demonstrates ingesting a sample PDF and running a test query.

```bash
python rag_mongodb.py
```

It will:
- Ingest the PDF defined in `PDF_URL` in `rag_mongodb.py`
- Create or verify the vector index
- Run an example query and print the final answer

---

## 💡 Tips & Troubleshooting

- Ensure the **embedding dimension** in `rag_mongodb.py` (`EMBEDDING_DIM`) matches the embedding model you use.
- If `create_vector_index()` fails, confirm your MongoDB deployment supports Search Indexes and vectorSearch.
- If the OpenAI embeddings call fails, verify `OPENAI_API_KEY` and check usage/quotas.
- Watch the console for messages about index readiness — the code waits until the index is `queryable`.

---

## 🛠️ Extending this project

- Add support for more loaders (web pages, DOCX, plaintext)
- Swap embeddings / LLM providers (Hugging Face, Azure OpenAI, etc.)
- Add a Streamlit UI for interactive querying
- Implement batching and bulk ingestion for large corpora

---

## 📄 Contributing

Contributions, issues, and feature requests are welcome — open a PR or an issue.

---

## 📜 License

MIT

---
