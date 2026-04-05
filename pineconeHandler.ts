import { Pinecone, QueryResponse } from '@pinecone-database/pinecone';
import config from './config.ts';

export type HybridVectors = {
  weightedDenseValues: number[];
  weightedSparseValues: number[];
  sparseIndices: number[];
};

export type ChunkResult = { id: string; text: string };

// Validate required configuration before initializing
const requiredConfigs: { key: string; value: string }[] = [
    { key: 'PINECONE_API_KEY', value: config.PINECONE_API_KEY },
    { key: 'INDEX_NAME', value: config.INDEX_NAME },
    { key: 'NAME_SPACE', value: config.NAME_SPACE },
    { key: 'DENSE_EMBED_MODEL', value: config.DENSE_EMBED_MODEL },
    { key: 'SPARSE_EMBED_MODEL', value: config.SPARSE_EMBED_MODEL },
];
const missingConfigs = requiredConfigs.filter(c => !c.value).map(c => c.key);
if (missingConfigs.length > 0) {
    throw new Error(`Missing required configuration: ${missingConfigs.join(', ')}. Please check your environment variables and config.`);
}

// Initialize a Pinecone client with your API key
const pc = new Pinecone({ apiKey: config.PINECONE_API_KEY });

// Target the index
const index = pc.index(config.INDEX_NAME).namespace(config.NAME_SPACE);

/**
 * Retry a function up to `maxRetries` times with exponential backoff.
 */
async function withRetry<T>(fn: () => Promise<T>, maxRetries: number = config.api.retries, baseDelayMs: number = 500): Promise<T> {
    let lastError: Error | unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (attempt < maxRetries) {
                const delay = baseDelayMs * Math.pow(2, attempt);
                console.warn(`Attempt ${attempt + 1} failed, retrying in ${delay}ms...`, error instanceof Error ? error.message : error);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }
    throw lastError;
}

export async function vectorize(query: string, alpha: number = config.HYBRID_SEARCH_ALPHA): Promise<HybridVectors> {
    // Adjust the weight between dense and sparse vectors.
    // alpha = 1.0 → pure dense (semantic) search
    // alpha = 0.0 → pure sparse (keyword) search
    // alpha = 0.5 → equal weight to both

    const denseEmbedding = await withRetry(() => pc.inference.embed({
        model: config.DENSE_EMBED_MODEL,
        inputs: [ query ],
        parameters: {
            inputType: 'query',
            truncate: 'END',
        }
    }));

    const sparseEmbedding = await withRetry(() => pc.inference.embed({
        model: config.SPARSE_EMBED_MODEL,
        inputs: [ query ],
        parameters: {
            inputType: 'query',
            truncate: 'END',
        }
    }));

    const denseVector = denseEmbedding.data[0];
    const sparseVector = sparseEmbedding.data[0];

    if (denseVector.vectorType !== 'dense') {
        throw new Error('Expected dense embedding');
    }

    if (sparseVector.vectorType !== 'sparse') {
        throw new Error('Expected sparse embedding');
    }

    const weightedDenseValues = denseVector.values.map((v: number) => v * alpha);
    const weightedSparseValues = sparseVector.sparseValues.map((v: number) => v * (1 - alpha));

    return {
        weightedDenseValues,
        weightedSparseValues,
        sparseIndices: sparseVector.sparseIndices
    };
}

export async function searchChunks(hybridVectors: HybridVectors): Promise<ChunkResult[]> {
    const searchResponse: QueryResponse = await withRetry(() => index.query({
        topK: config.TOP_K,
        vector: hybridVectors.weightedDenseValues,
        sparseVector: {
            indices: hybridVectors.sparseIndices,
            values: hybridVectors.weightedSparseValues,
        },
        includeMetadata: true,
        includeValues: false,
    }));

    const SCORE_THRESHOLD = 0.3;

    return searchResponse.matches
        .filter((match) => !!match.metadata && (match.score ?? 0) > SCORE_THRESHOLD)
        .map((match, i) => ({
            id: match.id,
            text: match.metadata?.chunk_text as string,
        }));
}




