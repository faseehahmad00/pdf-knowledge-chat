import express from 'express';
const router = express.Router();
import dotenv from 'dotenv/config';
import { loadPDF } from '../services/readingService.js';
import { generateEmbedding } from '../services/embeddingService.js';
import { splitText } from '../utils/utils.js';
import { getPineconeIndex, alreadyProcessed, saveChunksToPinecone } from '../services/pineconeService.js';
import { getGroqLLM, buildChatPrompt } from '../services/groqService.js';

router.get('/process-pdf', async (req, res) => {
  try {
    const text = await loadPDF();
    const chunks = splitText(text);
    const index = await getPineconeIndex();
    const exists = await alreadyProcessed(index);
    if (exists) {
      return res.send('Embeddings already exist in Pinecone. Skipping processing.');
    }
    await saveChunksToPinecone(index, chunks, generateEmbedding);
    res.send('PDF processed, embeddings stored successfully.');
  } catch (err) {
    console.error(`[Error]`, err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while processing the PDF.', details: err });
  }
});

router.post('/query', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query text is required.' });
    }
    const queryEmbedding = await generateEmbedding(query);
    const index = await getPineconeIndex();
    const results = await index.query({
      topK: 5,
      vector: queryEmbedding,
      includeMetadata: true
    });
    const relatedTexts = (results.matches || []).map(match => match.metadata?.text).filter(Boolean);
    res.json({ relatedTexts });
  } catch (err) {
    console.error('[Query Error]', err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while querying Pinecone.', details: err });
  }
});

router.post('/chat', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query text is required.' });
    }
    const queryEmbedding = await generateEmbedding(query);
    const index = await getPineconeIndex();
    const results = await index.query({
      topK: 10,
      vector: queryEmbedding,
      includeMetadata: true
    });
    const relatedTexts = (results.matches || []).map(match => match.metadata?.text).filter(Boolean);
    let context = relatedTexts.join("\n\n").slice(0, 12000);
    const prompt = buildChatPrompt(context);
    const llm = getGroqLLM();
    const chain = prompt.pipe(llm);
    const response = await chain.invoke({ input: query });
    res.json({ response: response.content });
  } catch (err) {
    console.error('[Chat Error]', err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while generating a chat response.', details: err });
  }
});

export default router;
