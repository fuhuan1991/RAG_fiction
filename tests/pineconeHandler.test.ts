import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ── Hoisted mock references (available inside vi.mock factories) ─────────────
const { mockEmbed, mockQuery } = vi.hoisted(() => ({
    mockEmbed: vi.fn(),
    mockQuery: vi.fn(),
}));

// ── Module mocks (hoisted above imports by vitest) ───────────────────────────

vi.mock('../config.ts', () => ({
    default: {
        PINECONE_API_KEY: 'test-api-key',
        INDEX_NAME: 'test-index',
        NAME_SPACE: 'test-namespace',
        DENSE_EMBED_MODEL: 'test-dense-model',
        SPARSE_EMBED_MODEL: 'test-sparse-model',
        OPENAI_API_KEY: 'test-openai-key',
        TOP_K: 10,
        HYBRID_SEARCH_ALPHA: 0.7,
        GPT_MODEL: 'gpt-4.1-mini',
        BOOK_NAME: 'Babel',
        RELEVANT_CHECK_MAX_TOOL_CALLS: 3,
        SEARCH_MAX_TOOL_CALLS: 10,
        MULTI_QUERY_COUNT: 3,
        QUERY_EXPANSION_MODEL: 'gpt-4.1-nano',
        env: 'test',
        server: { port: 3000, host: 'localhost' },
        api: { timeout: 30000, retries: 3 },
    },
}));

vi.mock('@pinecone-database/pinecone', () => ({
    Pinecone: class MockPinecone {
        inference = { embed: mockEmbed };
        index() {
            return {
                namespace() {
                    return { query: mockQuery };
                },
            };
        }
    },
}));

// Mock RunnableLambda as a pass-through so the inner logic is still executed
vi.mock('@langchain/core/runnables', () => ({
    RunnableLambda: {
        from: (fn: Function) => ({
            withConfig: () => ({
                invoke: (input: any) => fn(input),
            }),
        }),
    },
}));

// ── Import module under test (mocks are already in place) ────────────────────
import { vectorize, searchChunks, type HybridVectors } from '../pineconeHandler.ts';

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeDenseEmbeddingResponse(values: number[]) {
    return {
        data: [{ vectorType: 'dense', values }],
    };
}

function makeSparseEmbeddingResponse(values: number[], indices: number[]) {
    return {
        data: [{ vectorType: 'sparse', sparseValues: values, sparseIndices: indices }],
    };
}

/** Set up mockEmbed to return one dense + one sparse embedding in sequence. */
function setupEmbedMocks(
    denseValues: number[],
    sparseValues: number[],
    sparseIndices: number[],
) {
    mockEmbed
        .mockResolvedValueOnce(makeDenseEmbeddingResponse(denseValues))
        .mockResolvedValueOnce(makeSparseEmbeddingResponse(sparseValues, sparseIndices));
}

// ── Tests ────────────────────────────────────────────────────────────────────

beforeEach(() => {
    mockEmbed.mockReset();
    mockQuery.mockReset();
});

// ──────────────────────────────────────────────────────────────────────────────
//  vectorize()
// ──────────────────────────────────────────────────────────────────────────────

describe('vectorize', () => {
    it('applies the default alpha (0.7) weighting correctly', async () => {
        setupEmbedMocks([1.0, 2.0, 3.0], [4.0, 5.0], [10, 20]);

        const result = await vectorize('test query');

        expect(result.weightedDenseValues).toEqual([
            expect.closeTo(0.7), expect.closeTo(1.4), expect.closeTo(2.1),
        ]);
        expect(result.weightedSparseValues).toEqual([
            expect.closeTo(1.2), expect.closeTo(1.5),
        ]);
        expect(result.sparseIndices).toEqual([10, 20]);
    });

    it('applies alpha=1.0 (pure semantic) — sparse values are zeroed', async () => {
        setupEmbedMocks([1.0, 2.0], [4.0, 5.0], [10, 20]);

        const result = await vectorize('test', 1.0);

        expect(result.weightedDenseValues).toEqual([1.0, 2.0]);
        expect(result.weightedSparseValues).toEqual([0.0, 0.0]);
    });

    it('applies alpha=0.0 (pure keyword) — dense values are zeroed', async () => {
        setupEmbedMocks([1.0, 2.0], [4.0, 5.0], [10, 20]);

        const result = await vectorize('test', 0.0);

        expect(result.weightedDenseValues).toEqual([0.0, 0.0]);
        expect(result.weightedSparseValues).toEqual([4.0, 5.0]);
    });

    it('applies alpha=0.5 (equal weight) — both halved', async () => {
        setupEmbedMocks([2.0, 4.0], [6.0, 8.0], [1, 2]);

        const result = await vectorize('test', 0.5);

        expect(result.weightedDenseValues).toEqual([1.0, 2.0]);
        expect(result.weightedSparseValues).toEqual([3.0, 4.0]);
    });

    it('passes correct parameters to Pinecone embed API', async () => {
        setupEmbedMocks([1.0], [1.0], [0]);

        await vectorize('hello world');

        // First call — dense embedding
        expect(mockEmbed).toHaveBeenNthCalledWith(1, {
            model: 'test-dense-model',
            inputs: ['hello world'],
            parameters: { inputType: 'query', truncate: 'END' },
        });

        // Second call — sparse embedding
        expect(mockEmbed).toHaveBeenNthCalledWith(2, {
            model: 'test-sparse-model',
            inputs: ['hello world'],
            parameters: { inputType: 'query', truncate: 'END' },
        });
    });

    it('throws when dense embedding returns wrong vectorType', async () => {
        // Both embeds are awaited before vectorType is checked, so mock both
        mockEmbed
            .mockResolvedValueOnce({
                data: [{ vectorType: 'sparse', sparseValues: [], sparseIndices: [] }],
            })
            .mockResolvedValueOnce(makeSparseEmbeddingResponse([1.0], [0]));

        await expect(vectorize('test')).rejects.toThrow('Expected dense embedding');
    });

    it('throws when sparse embedding returns wrong vectorType', async () => {
        mockEmbed
            .mockResolvedValueOnce(makeDenseEmbeddingResponse([1.0]))
            .mockResolvedValueOnce({
                data: [{ vectorType: 'dense', values: [1.0] }],
            });

        await expect(vectorize('test')).rejects.toThrow('Expected sparse embedding');
    });
});

// ──────────────────────────────────────────────────────────────────────────────
//  searchChunks()
// ──────────────────────────────────────────────────────────────────────────────

describe('searchChunks', () => {
    const sampleInput: HybridVectors = {
        weightedDenseValues: [0.7, 1.4],
        weightedSparseValues: [1.2, 1.5],
        sparseIndices: [10, 20],
    };

    it('returns chunks above score threshold with correct format', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'chunk-1', score: 0.8, metadata: { chunk_text: 'Hello world' } },
                { id: 'chunk-2', score: 0.5, metadata: { chunk_text: 'Foo bar' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([
            { id: 'chunk-1', text: 'Hello world' },
            { id: 'chunk-2', text: 'Foo bar' },
        ]);
    });

    it('filters out matches with score ≤ 0.3', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'high', score: 0.9, metadata: { chunk_text: 'keep' } },
                { id: 'low', score: 0.2, metadata: { chunk_text: 'discard' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([{ id: 'high', text: 'keep' }]);
    });

    it('excludes score exactly equal to 0.3 (threshold is exclusive)', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'boundary', score: 0.3, metadata: { chunk_text: 'on the line' } },
                { id: 'above', score: 0.31, metadata: { chunk_text: 'just above' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([{ id: 'above', text: 'just above' }]);
    });

    it('filters out matches without metadata', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'no-meta', score: 0.9, metadata: null },
                { id: 'with-meta', score: 0.9, metadata: { chunk_text: 'valid' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([{ id: 'with-meta', text: 'valid' }]);
    });

    it('filters out matches with undefined metadata', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'undef', score: 0.9 },
                { id: 'ok', score: 0.9, metadata: { chunk_text: 'present' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([{ id: 'ok', text: 'present' }]);
    });

    it('treats undefined score as 0 (filtered out)', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'no-score', metadata: { chunk_text: 'missing score' } },
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([]);
    });

    it('returns empty array when there are no matches', async () => {
        mockQuery.mockResolvedValue({ matches: [] });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([]);
    });

    it('correctly handles a mix of valid and invalid matches', async () => {
        mockQuery.mockResolvedValue({
            matches: [
                { id: 'a', score: 0.9, metadata: { chunk_text: 'good' } },        // ✓ pass
                { id: 'b', score: 0.1, metadata: { chunk_text: 'low score' } },    // ✗ score too low
                { id: 'c', score: 0.8, metadata: null },                            // ✗ no metadata
                { id: 'd', score: 0.7, metadata: { chunk_text: 'also good' } },    // ✓ pass
                { id: 'e', score: 0.3, metadata: { chunk_text: 'boundary' } },     // ✗ exactly 0.3
                { id: 'f', score: 0.5 },                                            // ✗ undefined metadata
            ],
        });

        const result = await searchChunks(sampleInput);

        expect(result).toEqual([
            { id: 'a', text: 'good' },
            { id: 'd', text: 'also good' },
        ]);
    });

    it('passes correct query parameters to Pinecone', async () => {
        mockQuery.mockResolvedValue({ matches: [] });

        const input: HybridVectors = {
            weightedDenseValues: [0.1, 0.2],
            weightedSparseValues: [0.3, 0.4],
            sparseIndices: [5, 15],
        };

        await searchChunks(input);

        expect(mockQuery).toHaveBeenCalledWith({
            topK: 10,
            vector: [0.1, 0.2],
            sparseVector: {
                indices: [5, 15],
                values: [0.3, 0.4],
            },
            includeMetadata: true,
            includeValues: false,
        });
    });
});

// ──────────────────────────────────────────────────────────────────────────────
//  withRetry (tested indirectly through vectorize)
// ──────────────────────────────────────────────────────────────────────────────

describe('withRetry (via vectorize)', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('succeeds on first attempt without needing retries', async () => {
        setupEmbedMocks([1.0], [2.0], [0]);

        const result = await vectorize('no retry needed');

        expect(result.weightedDenseValues).toEqual([0.7]);
        expect(mockEmbed).toHaveBeenCalledTimes(2); // 1 dense + 1 sparse, no retries
    });

    it('retries on transient failure and eventually succeeds', async () => {
        // Dense embed: fail once, then succeed. Sparse embed: succeed immediately.
        mockEmbed
            .mockRejectedValueOnce(new Error('transient network error'))
            .mockResolvedValueOnce(makeDenseEmbeddingResponse([1.0]))
            .mockResolvedValueOnce(makeSparseEmbeddingResponse([2.0], [0]));

        const promise = vectorize('retry test');
        await vi.runAllTimersAsync();
        const result = await promise;

        expect(result.weightedDenseValues).toEqual([0.7]);
        expect(result.weightedSparseValues).toEqual([0.6000000000000001]);
        // 3 calls total: 1 failed dense + 1 successful dense retry + 1 sparse
        expect(mockEmbed).toHaveBeenCalledTimes(3);
    });

    it('throws the last error after exhausting all retries', async () => {
        // All embed calls fail — withRetry will exhaust retries (config.api.retries = 3)
        // Use mockRejectedValueOnce × 4 to avoid stray unhandled rejections
        mockEmbed
            .mockRejectedValueOnce(new Error('persistent failure'))
            .mockRejectedValueOnce(new Error('persistent failure'))
            .mockRejectedValueOnce(new Error('persistent failure'))
            .mockRejectedValueOnce(new Error('persistent failure'));

        // Attach .catch immediately to prevent unhandled rejection warnings
        let caughtError: unknown;
        const promise = vectorize('doomed query').catch(e => { caughtError = e; });
        await vi.runAllTimersAsync();
        await promise;

        expect(caughtError).toBeInstanceOf(Error);
        expect((caughtError as Error).message).toBe('persistent failure');
        // 4 total attempts for the first embed call: 1 initial + 3 retries
        expect(mockEmbed).toHaveBeenCalledTimes(4);
    });
});

// ──────────────────────────────────────────────────────────────────────────────
//  Module-level config validation
// ──────────────────────────────────────────────────────────────────────────────

describe('config validation', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('throws an error listing all missing required config keys', async () => {
        vi.resetModules();

        vi.doMock('../config.ts', () => ({
            default: {
                PINECONE_API_KEY: '',
                INDEX_NAME: '',
                NAME_SPACE: 'test-namespace',
                DENSE_EMBED_MODEL: 'test-dense-model',
                SPARSE_EMBED_MODEL: '',
                TOP_K: 10,
                HYBRID_SEARCH_ALPHA: 0.7,
                api: { timeout: 30000, retries: 3 },
            },
        }));

        // Re-mock external deps so the module can still parse
        vi.doMock('@pinecone-database/pinecone', () => ({
            Pinecone: vi.fn().mockImplementation(() => ({
                inference: { embed: vi.fn() },
                index: vi.fn().mockReturnValue({
                    namespace: vi.fn().mockReturnValue({ query: vi.fn() }),
                }),
            })),
        }));
        vi.doMock('@langchain/core/runnables', () => ({
            RunnableLambda: {
                from: (fn: Function) => ({
                    withConfig: () => ({ invoke: (input: any) => fn(input) }),
                }),
            },
        }));

        await expect(() => import('../pineconeHandler.ts')).rejects.toThrow(
            'Missing required configuration: PINECONE_API_KEY, INDEX_NAME, SPARSE_EMBED_MODEL'
        );
    });
});
