import { ChatOpenAI } from '@langchain/openai';
import config from '../config.ts';
import { vectorize, searchChunks, HybridVectors, ChunkResult } from '../pineconeHandler.ts';
import { queryExpansionPromptTemplate, queryExpansionOutput } from '../promptTemplates/queryExpansionPrompt.ts';

export const GET_INTERNAL_INFO: string = 'get_internal_info';
const EXPAND_QUERY_TEMPERATURE = 0.7; // moderate creativity for diverse phrasings

/**
 * Search the novel's content in Pinecone.
 */
async function internalSearch(query: string): Promise<ChunkResult[]> {
    const hybridVectors: HybridVectors = await vectorize(query);
    const chunks: ChunkResult[] = await searchChunks(hybridVectors);
    return chunks;
}

/**
 * Use an LLM to generate alternative phrasings of the original query
 * to improve vector search recall (multi-query expansion).
 * Returns the original query + N-1 alternatives.
 */
async function expandQuery(originalQuery: string): Promise<string[]> {
    const alternativeCount = config.MULTI_QUERY_COUNT - 1;

    // If only 1 query is configured, skip expansion entirely
    if (alternativeCount <= 0) {
        return [originalQuery];
    }

    try {
        const llm = new ChatOpenAI({
            model: config.QUERY_EXPANSION_MODEL,
            temperature: EXPAND_QUERY_TEMPERATURE,
            apiKey: config.OPENAI_API_KEY,
        });

        const structuredLlm = llm.withStructuredOutput(queryExpansionOutput);

        const queryExpansionPrompt = await queryExpansionPromptTemplate.formatMessages({
            originalQuery,
            alternativeCount: String(alternativeCount),
        });

        const result = await structuredLlm.invoke(queryExpansionPrompt, { runName: "query_expansion_llm_call" });

        // Combine original + alternatives, ensuring we don't exceed the configured count
        const alternatives = result.alternatives.slice(0, alternativeCount);

        return [originalQuery, ...alternatives];
    } catch (error) {
        // If expansion fails, fall back to just the original query
        return [originalQuery];
    }
}

export async function internalSearchTool (neededInformation: string, seenChunkIds: Set<string> ) {
    try {
        // Step 1: Expand the query into multiple alternative phrasings
        const expandedQueries = await expandQuery(neededInformation);

        // Step 2: Run vector searches in parallel for all expanded queries
        const searchResults = await Promise.all(
            expandedQueries.map(query => internalSearch(query))
        );

        // Step 3: Merge and deduplicate results across all expanded queries
        const newChunks: ChunkResult[] = [];
        for (const chunks of searchResults) {
            for (const chunk of chunks) {
                if (!seenChunkIds.has(chunk.id)) {
                    seenChunkIds.add(chunk.id);
                    newChunks.push(chunk);
                }
            }
        }

        return newChunks;
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Failed to search the novel's text: ${errorMessage}. Please try again or rephrase the query.`);
    }
}
