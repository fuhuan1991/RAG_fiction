import * as z from 'zod';
import { tool } from 'langchain';
import config from '../config.ts';
import { vectorize, searchChunks, HybridVectors } from '../pineconeHandler.ts';

export const GET_INTERNAL_INFO: string = 'get_internal_info';

/**
 * Search the novel's content in Pinecone.
 */
async function internalSearch(query: string): Promise<string[]> {
    const hybridVectors: HybridVectors = await vectorize(query);
    const chunks: string[] = await searchChunks(hybridVectors);
    return chunks;
}

export const getInternalInfo = tool(
    async ({ infoRequests }) => {

        const existingChunks: string[] = [];

        console.log('The following internal information is needed:');
        for (const item of infoRequests) {
            console.log('-' + item.neededInformation);
        }

        await Promise.all(
            infoRequests.map(async (request) => {
                const { neededInformation } = request;
                const chunks = await internalSearch(neededInformation);
                const newChunks = chunks.filter(chunk => !existingChunks.includes(chunk));
                existingChunks.push(...newChunks);
            })
        );

        return existingChunks;
    },
    {
        name: GET_INTERNAL_INFO,
        description:
            `Search through the novel "${config.BOOK_NAME}" for internal information such as plot details, character info, events, themes, quotes, etc. ` +
            `This tool accepts multiple information requests at once, each specifying what information is needed. ` +
            `It searches the novel's text in a vector database using embedding-based similarity and returns an array of relevant text chunks. ` +
            `Query optimization guidelines for the "neededInformation" field: ` +
            `1) Use natural language, not keywords — write descriptive sentences or phrases, not boolean operators or comma-separated keywords. For example, instead of "protagonist, motivation, backstory", write "the protagonist's motivation and backstory including key events that shaped their personality". ` +
            `2) Be specific and include proper nouns — use character names, place names, and specific event descriptions. For example, instead of "the battle scene", write "the battle at [location] where [character] confronts [character]". ` +
            `3) Match the language style of the novel — phrase the query in a way that resembles how the novel's text would describe the topic, so the embedding similarity is maximized. ` +
            `4) One focused topic per query — each query should target a single, specific piece of information. Broad queries dilute search relevance. ` +
            `5) Avoid meta-language — do not write "I need information about..." or "Find passages related to...". Instead, directly state the content you are looking for, e.g., "Elizabeth's feelings toward Darcy after the letter revelation".`,
        schema: z.object({
            infoRequests: z.array(
                z.object({
                    neededInformation: z.string().describe(
                        'A natural-language search query used to retrieve relevant passages from the novel via vector search. ' +
                        'Write this as a semantically rich, descriptive statement that closely mirrors how the desired information would be expressed in the novel\'s text. ' +
                        'Be specific, use proper nouns, and focus on a single topic per query.'
                    ),
                })
            ).describe(
                'An array of information requests. Each request specifies what information is needed from the novel.'
            )
        }),
    }
);
