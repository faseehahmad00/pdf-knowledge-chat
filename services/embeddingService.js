import axios from 'axios';

export async function generateEmbedding(text) {
  if (!text.trim()) {
    throw new Error('Chunk is empty or whitespace');
  }
  console.log(`[Embedding] Generating embedding...`);
  const response = await axios.post(
    'http://localhost:8000/embed',
    { inputs: text },
    {
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY}`
      }
    }
  );
  console.log(`[Embedding] Embedding generated`);
  return response.data.embeddings[0];
} 