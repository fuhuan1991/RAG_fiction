import * as z from 'zod';
import { tool } from 'langchain';
import config from '../config.ts';
import { vectorize, searchChunks, HybridVectors } from '../pineconeHandler.ts';

export const NOVEL_CONTENT_SEARCH:string = 'novel_content_search';

export const novelContentSearch = tool(
    async ({ query }) => {
        const hybridVectors: HybridVectors = await vectorize(query);
        const chunks: string[] = await searchChunks(hybridVectors);

        console.log('----novelContentSearch result:');
        console.log(chunks);
        return chunks;
    },
    {
        name: NOVEL_CONTENT_SEARCH,
        description:
            `Search through the text of the fictional novel "${config.BOOK_NAME}" to find relevant passages. ` +
            `This tool performs a hybrid semantic and keyword search against a vector database that contains chunked text from the novel. ` +
            `Use this tool whenever the user asks questions about the plot, characters, events, themes, quotes, or any other details from "${config.BOOK_NAME}". ` +
            `The tool returns the most relevant text passages from the novel that match the query. ` +
            `Do NOT use this tool for questions unrelated to the novel "${config.BOOK_NAME}".`,
        schema: z.object({
            query: z.string().describe(
                'A natural language search query describing the information you want to find from the novel. ' +
                'Be specific and descriptive. For example, use character names, events, or themes as keywords. ' +
                'If the user\'s question is complex, break it down and search for the most relevant aspect.'
            )
        }),
    }
);