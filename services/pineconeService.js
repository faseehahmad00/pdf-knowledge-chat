import { Pinecone } from '@pinecone-database/pinecone';

// Cache for storing fetched indexes
const indexCache = new Map();

export async function getPineconeIndex(indexName = 'manual-embeddings') {
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const result = await pinecone.listIndexes();
  if (!result.indexes.includes(indexName)) {
    try {
      await pinecone.createIndex({
        name: indexName,
        dimension: 384,
        metric: 'cosine',
        spec: {
          serverless: {
            cloud: 'aws',
            region: 'us-east-1',
          }
        }
      });
    } catch (err) {
      if (
        (err.response &&
          err.response.status === 409 &&
          err.response.data?.error?.code === 'ALREADY_EXISTS') ||
        err.name === 'PineconeConflictError'
      ) {
        console.log(`[Pinecone] Index already exists, continuing...`);
      } else {
        throw err;
      }
    }
  }
  return pinecone.Index(indexName);
}

// New function to get cached index
export async function getCachedIndex(indexName = 'manual-embeddings') {
  // Check if index is already cached
  if (indexCache.has(indexName)) {
    console.log(`[Cache HIT] Using cached index: ${indexName}`);
    return indexCache.get(indexName);
  }

  // If not cached, get the index and cache it
  console.log(`[Cache MISS] Fetching and caching index: ${indexName}`);
  const index = await getPineconeIndex(indexName);
  indexCache.set(indexName, index);
  
  return index;
}

// Utility functions for cache management
export function clearIndexCache(indexName) {
  if (indexCache.has(indexName)) {
    indexCache.delete(indexName);
    console.log(`[Cache] Cleared cache for index: ${indexName}`);
    return true;
  }
  return false;
}

export function clearAllIndexCache() {
  const size = indexCache.size;
  indexCache.clear();
  console.log(`[Cache] Cleared all index cache (${size} indexes)`);
}

export function getCacheStats() {
  return {
    cachedIndexes: Array.from(indexCache.keys()),
    cacheSize: indexCache.size
  };
}

export function isIndexCached(indexName) {
  return indexCache.has(indexName);
}

export async function alreadyProcessed(index) {
  const stats = await index.describeIndexStats();
  const totalVectors = stats.totalVectorCount || 0;
  return totalVectors > 0;
}

export async function saveChunksToPinecone(index, chunks, generateEmbedding) {
  const vectors = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = await generateEmbedding(chunk);
    vectors.push({
      id: `manual-chunk-${i}`,
      values: embedding,
      metadata: { text: chunk }
    });
  }
  await index.upsert(vectors);
} 



