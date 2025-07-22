import { Pinecone } from '@pinecone-database/pinecone';

export async function getPineconeIndex() {
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const indexName = 'manual-embeddings';
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