import { Pinecone, QueryResponse } from '@pinecone-database/pinecone';
import config from './config.ts';

export type HybridVectors = {
  weightedDenseValues: number[];
  weightedSparseValues: number[];
  sparseIndices: number[];
};

// Initialize a Pinecone client with your API key
const pc = new Pinecone({ apiKey: config.PINECONE_API_KEY });

// Target the index
const index = pc.index(config.INDEX_NAME).namespace(config.NAME_SPACE);

export async function vectorize(query: string, alpha: number = 0.7): Promise<HybridVectors> {
    // Adjust the weight between dense and sparse vectors.
    // alpha = 1.0 → pure dense (semantic) search
    // alpha = 0.0 → pure sparse (keyword) search
    // alpha = 0.5 → equal weight to both

    const denseEmbedding = await pc.inference.embed({
        model: config.DENSE_EMBED_MODEL,
        inputs: [ query ],
        parameters: {
            inputType: 'passage',
            truncate: 'END',
        }
    });

    const sparseEmbedding = await pc.inference.embed({
        model: config.SPARSE_EMBED_MODEL,
        inputs: [ query ],
        parameters: {
            inputType: 'passage',
            truncate: 'END',
        }
    });

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

export async function searchChunks(hybridVectors: HybridVectors): Promise<string[]> {
    const searchResponse: QueryResponse = await index.query({
        topK: config.TOP_K,
        vector: hybridVectors.weightedDenseValues,
        sparseVector: {
            indices: hybridVectors.sparseIndices,
            values: hybridVectors.weightedSparseValues,
        },
        includeMetadata: true,
        includeValues: false,
    });

    const chunks: string[] = searchResponse.matches
    .filter((match) => !!match.metadata)
    .map((match, i) => {
        const text = match.metadata?.chunk_text;
        return `   ${text}`;
    });

    return chunks;
}




