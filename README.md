# 🦙 RAG-AI: PDF Question Answering with Pinecone, Local Embeddings, and Groq Llama 3

## Overview

**RAG-AI** is a modern Retrieval-Augmented Generation (RAG) system that lets you upload a PDF, index its content as vector embeddings, and then query it using natural language. The system retrieves the most relevant chunks from your PDF using Pinecone, and generates accurate, context-aware answers using Groq’s blazing-fast Llama 3 70B model.

- **PDF Parsing**: Extracts and splits your PDF into manageable text chunks.
- **Local Embeddings**: Generates embeddings for each chunk using a local Sentence Transformers server (no paid API required!).
- **Vector Database**: Stores and searches embeddings in Pinecone for fast, scalable retrieval.
- **LLM-Powered Answers**: Uses Groq’s Llama 3 70B via LangChain to generate answers based on retrieved context.
- **Modular & Clean**: All logic is separated into services for easy maintenance and extension.

---

## Features

- **/process-pdf**: Parse and index your PDF into Pinecone.
- **/query**: Retrieve the most relevant text chunks for a user question.
- **/chat**: Get a full LLM-generated answer, grounded in your PDF, using Groq Llama 3.
- **Local Embedding Server**: No OpenAI/HuggingFace API costs—run your own embedding server with Python.
- **Clean, Modern Codebase**: All business logic is modularized in `/services`.

---

## Architecture

```
User Query
   │
   ▼
[Express API]
   │
   ├── /process-pdf ──► [PDF Parser] ──► [Local Embedding Server] ──► [Pinecone Vector DB]
   │
   ├── /query ──► [Embedding] ──► [Pinecone Search] ──► [Top Chunks]
   │
   └── /chat ──► [Embedding] ──► [Pinecone Search] ──► [Top Chunks] ──► [Groq Llama 3 LLM]
```

---

## How It Works

### 1. PDF Processing (`/process-pdf`)
- Reads and parses a PDF (`public/manual.pdf` by default).
- Splits the text into overlapping chunks.
- Generates embeddings for each chunk using a local Python FastAPI server running `sentence-transformers`.
- Stores all embeddings and their text in Pinecone.

### 2. Querying (`/query`)
- Accepts a user question.
- Embeds the question using the same local embedding server.
- Searches Pinecone for the top 5 most similar chunks.
- Returns the related text chunks.

### 3. Chat (`/chat`)
- Accepts a user question.
- Embeds the question and retrieves the top 10 relevant chunks from Pinecone.
- Concatenates the context (up to Llama 3’s context window).
- Sends the context and question to Groq’s Llama 3 70B model via LangChain.
- Returns the LLM’s answer, grounded in your PDF.

---

## Directory Structure

```
.
├── app.js
├── routes/
│   └── index.js
├── services/
│   ├── embeddingService.js
│   ├── groqService.js
│   ├── pineconeService.js
│   ├── readingService.js
│   └── utils.js
├── public/
│   └── manual.pdf
├── .env
└── ...
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-ai
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Set Up Environment Variables

Create a `.env` file in the root:

```env
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
HF_API_KEY=your-huggingface-api-key-or-any-string
```

### 4. Set Up the Local Embedding Server

You need Python 3.8+ and pip.

```bash
pip install fastapi uvicorn sentence-transformers
```

Create `embedding_server.py`:

```python
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/embed")
async def embed(request: Request):
    data = await request.json()
    sentences = data['inputs']
    if isinstance(sentences, str):
        sentences = [sentences]
    embeddings = model.encode(sentences).tolist()
    return {"embeddings": embeddings}
```

Run the server:

```bash
python -m uvicorn embedding_server:app --host 0.0.0.0 --port 8000
```

### 5. Add Your PDF

Place your PDF as `public/manual.pdf`.

### 6. Start the Node.js Server

```bash
npm start
```

---

## API Endpoints

### `GET /process-pdf`
- Parses and indexes the PDF into Pinecone.
- Only needs to be run once per PDF.

### `POST /query`
- **Body:** `{ "query": "your question" }`
- **Returns:** Top 5 related text chunks.

### `POST /chat`
- **Body:** `{ "query": "your question" }`
- **Returns:** LLM-generated answer, grounded in your PDF.

---

## Customization

- **Change PDF:** Replace `public/manual.pdf` with your own document.
- **Change chunk size/overlap:** Edit `splitText` in `services/utils.js`.
- **Change embedding model:** Update the model in your Python server.
- **Change LLM model:** Update the model name in `services/groqService.js`.

---

## Credits

- [Pinecone](https://www.pinecone.io/) for vector search.
- [Groq](https://groq.com/) for ultra-fast Llama 3 inference.
- [LangChain](https://js.langchain.com/) for LLM orchestration.
- [Sentence Transformers](https://www.sbert.net/) for local embeddings.

---

## License

MIT

---

**Enjoy your private, blazing-fast, and cost-effective RAG system!**  
Questions or issues? Open an issue or PR! 