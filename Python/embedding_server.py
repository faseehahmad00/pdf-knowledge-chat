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
