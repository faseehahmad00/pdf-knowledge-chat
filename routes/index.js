import express from 'express';
const router = express.Router();
import dotenv from 'dotenv/config';
import { loadPDF } from '../services/readingService.js';
import { generateEmbedding } from '../services/embeddingService.js';
import { splitText } from '../utils/utils.js';
import { getCachedIndex, alreadyProcessed, saveChunksToPinecone } from '../services/pineconeService.js';
import { getAnswerFromContext, summarizeAndRephrase } from '../services/groqService.js';
import { traceable } from "langsmith/traceable";
import { Client } from "langsmith";

const client = new Client();

// Llama 3 70B context window: 8192 tokens ≈ 32,000 characters
const MODEL_CONTEXT_CHARS = 22000; //to keep groq's limit
const CHUNK_SIZE = 1000; // Should match your splitText default

router.get('/process-pdf', async (req, res) => {
  try {
    const indexName = req.query.indexName || 'manual-embeddings';
    const text = await loadPDF();
    const chunks = splitText(text);
    const index = await getCachedIndex(indexName); // Use cached index
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
    const index = await getCachedIndex(indexName); // Use cached index
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

router.post('/chat', traceable(async (req, res) => {
  try {
    const { query, indexName = 'manual-embeddings' } = req.body;
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query text is required.' });
    }

    // Trace embedding generation
    const queryEmbedding = await traceable(
      async (text) => await generateEmbedding(text),
      { name: "generate_embedding", tags: ["embedding"] }
    )(query);

    // Trace Pinecone operations
    const index = await traceable(
      async (name) => await getCachedIndex(name),
      { name: "get_pinecone_index", tags: ["pinecone"] }
    )(indexName);

    // Dynamically calculate topK based on model context window and chunk size
    const maxChunks = Math.floor(MODEL_CONTEXT_CHARS / CHUNK_SIZE);
    console.log("maxChunks", {maxChunks});

    // Trace vector search
    const results = await traceable(
      async ({ vector, topK }) => {
        return await index.query({
          topK,
          vector,
          includeMetadata: true
        });
      },
      { name: "pinecone_search", tags: ["pinecone", "search"] }
    )({ vector: queryEmbedding, topK: maxChunks });

    const relatedTexts = (results.matches || []).map(match => match.metadata?.text).filter(Boolean);
    let context = relatedTexts.join("\n\n").slice(0, MODEL_CONTEXT_CHARS);

    // Trace LLM answer generation
    const answer = await traceable(
      async ({ context, query }) => await getAnswerFromContext(context, query),
      { 
        name: "get_answer_from_context", 
        tags: ["llm", "groq"],
        metadata: { 
          context_length: context.length, 
          chunks_used: relatedTexts.length 
        }
      }
    )({ context, query });

    // Trace summarization
    const summarized = await traceable(
      async (text) => await summarizeAndRephrase(text),
      { name: "summarize_and_rephrase", tags: ["llm", "groq", "summarization"] }
    )(answer);

    res.json({ response: summarized });
  } catch (err) {
    console.error('[Chat Error]', err.response?.data || err.message);
    res.status(500).json({ error: 'An error occurred while generating a chat response.', details: err });
  }
}, {
  name: "chat_endpoint",
  tags: ["api", "chat"],
  metadata: { endpoint: "/chat" }
}));

export default router;
