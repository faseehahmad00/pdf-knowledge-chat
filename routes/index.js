import express from 'express';
const router = express.Router();
import dotenv from 'dotenv/config';
import { loadPDF } from '../services/readingService.js';
import { generateEmbedding } from '../services/embeddingService.js';
import { splitText } from '../utils/utils.js';
import { getPineconeIndex, alreadyProcessed, saveChunksToPinecone } from '../services/pineconeService.js';
import { getAnswerFromContext, summarizeAndRephrase } from '../services/groqService.js';

// Llama 3 70B context window: 8192 tokens ≈ 32,000 characters
const MODEL_CONTEXT_CHARS = 22000; //to keep groq's limit
const CHUNK_SIZE = 1000; // Should match your splitText default

router.get('/process-pdf', async (req, res) => {
  try {
    const indexName = req.query.indexName || 'manual-embeddings';
    const text = await loadPDF();
    const chunks = splitText(text);
    const index = await getPineconeIndex(indexName);
    const exists = await alreadyProcessed(index);
    if (exists) {
      return res.send(`Embeddings already exist in Pinecone for index ${indexName}. Skipping processing.`);
    }
    await saveChunksToPinecone(index, chunks, generateEmbedding);
    res.send(`PDF processed, embeddings stored successfully in index ${indexName}.`);
  } catch (err) {
    console.error(`[Error]`, err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while processing the PDF.', details: err });
  }
});

router.post('/query', async (req, res) => {
  try {
    const { query, indexName = 'manual-embeddings' } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query text is required.' });
    }
    const queryEmbedding = await generateEmbedding(query);
    const index = await getPineconeIndex(indexName);
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
    const { query, indexName = 'manual-embeddings' } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query text is required.' });
    }
    const queryEmbedding = await generateEmbedding(query);
    const index = await getPineconeIndex(indexName);
    // Dynamically calculate topK based on model context window and chunk size
    const maxChunks = Math.floor(MODEL_CONTEXT_CHARS / CHUNK_SIZE);
    console.log("maxChunks", {maxChunks});
    const results = await index.query({
      topK: maxChunks,  //to optimize the request time
      vector: queryEmbedding,
      includeMetadata: true
    });
    const relatedTexts = (results.matches || []).map(match => match.metadata?.text).filter(Boolean);
    let context = relatedTexts.join("\n\n").slice(0, MODEL_CONTEXT_CHARS);
    const answer = await getAnswerFromContext(context, query);
    const summarized = await summarizeAndRephrase(answer);
    res.json({ response: summarized });
  } catch (err) {
    console.error('[Chat Error]', err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while generating a chat response.', details: err });
  }
});

export default router;
