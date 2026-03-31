import * as z from 'zod';
import { tool } from 'langchain';
import config from '../config.ts';
import { vectorize, searchChunks, HybridVectors, ChunkResult } from '../pineconeHandler.ts';

export const GET_INTERNAL_INFO: string = 'get_internal_info';

/**
 * Search the novel's content in Pinecone.
 */
async function internalSearch(query: string): Promise<ChunkResult[]> {
    const hybridVectors: HybridVectors = await vectorize(query);
    const chunks: ChunkResult[] = await searchChunks(hybridVectors);
    return chunks;
}

/**
 * Factory function that creates a new getInternalInfo tool instance.
 * Each instance maintains its own set of seen chunk IDs to deduplicate
 * results across multiple calls within the same session.
 * 
 * Usage: Call this once per user request/session so that deduplication
 * state is not shared across different users.
 */
export function createInternalInfoTool() {
    // Per-session set to track chunk IDs that have already been returned
    const seenChunkIds = new Set<string>();
    const acquiredChunks: string[] = [];

    const newTool = tool(
        async ({ neededInformation }) => {

            console.log('----LLM wants to know this from the book: ' + neededInformation);

            try {
                const chunks: ChunkResult[] = await internalSearch(neededInformation);

                // Remove chunks that have already been returned in previous calls within this session
                const newChunks = chunks.filter(chunk => !seenChunkIds.has(chunk.id));

                // Track the new chunk IDs for future deduplication
                newChunks.forEach((chunk: ChunkResult) => {
                    acquiredChunks.push(chunk.text);
                    seenChunkIds.add(chunk.id)
                });

                console.log("----seenChunkIds length:" + seenChunkIds.size);
                console.log("----seenChunkIds:" + JSON.stringify([...seenChunkIds]));

                if (newChunks.length === 0) {
                    return 'No relevant passages found in the novel for this query.';
                }

                const outputForLLM = newChunks.map((chunk: ChunkResult) => {
                    return chunk.text;
                });

                return JSON.stringify(outputForLLM);
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                console.error('----internalSearch failed:', errorMessage);
                return `Failed to search the novel's text: ${errorMessage}. Please try again or rephrase the query.`;
            }
        },
        {
            name: GET_INTERNAL_INFO,
            description:
                `Search through the novel "${config.BOOK_NAME}" for internal information such as plot details, character info, events, themes, quotes, etc. ` +
                `It searches the novel's text in a vector database using embedding-based similarity and returns an array of relevant text chunks. ` +
                `Query optimization guidelines for the "neededInformation" parameter: ` +
                `1) Use natural language, not keywords — write descriptive sentences or phrases, not boolean operators or comma-separated keywords. For example, instead of "protagonist, motivation, backstory", write "the protagonist's motivation and backstory including key events that shaped their personality". ` +
                `2) Be specific and include proper nouns — use character names, place names, and specific event descriptions. For example, instead of "the battle scene", write "the battle at [location] where [character] confronts [character]". ` +
                `3) Match the language style of the novel — phrase the query in a way that resembles how the novel's text would describe the topic, so the embedding similarity is maximized. ` +
                `4) One focused topic per query — each query should target a single, specific piece of information. Broad queries dilute search relevance. ` +
                `5) Avoid meta-language — do not write "I need information about..." or "Find passages related to...". Instead, directly state the content you are looking for, e.g., "Elizabeth's feelings toward Darcy after the letter revelation".`,
            schema: z.object({
                neededInformation: z.string().describe(
                    'A natural-language search query used to retrieve relevant passages from the novel via vector search. ' +
                    'Write this as a semantically rich, descriptive statement that closely mirrors how the desired information would be expressed in the novel\'s text. ' +
                    'Be specific, use proper nouns, and focus on a single topic per query.'
                ),
            }),
        }
    );

    return [newTool, acquiredChunks] as const;
}
