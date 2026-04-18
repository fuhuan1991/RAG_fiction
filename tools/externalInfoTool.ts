import * as z from 'zod';
import { tool } from 'langchain';
import { RunnableLambda } from '@langchain/core/runnables';
import OpenAI from 'openai';
import config from '../config.ts';

export const GET_EXTERNAL_INFO: string = 'get_external_info';

const openai = new OpenAI({ apiKey: config.OPENAI_API_KEY });

/**
 * Perform a web search using OpenAI's Responses API with web_search_preview.
 * Wrapped in a RunnableLambda so the call is traced in LangSmith.
 */
const externalSearchRunnable = RunnableLambda.from(async (query: string): Promise<string> => {

    const response = await openai.responses.create({
        model: config.GPT_MODEL,
        tools: [{ type: 'web_search_preview' as const }],
        input: query,
    });

    // Extract text content from the response output
    const textParts: string[] = response.output
        .filter((item: any) => item.type === 'message')
        .flatMap((item: any) => item.content)
        .filter((content: any) => content.type === 'output_text')
        .map((content: any) => content.text);

    return textParts.join('\n') || 'No results found.';
}).withConfig({
    runName: 'external_search_llm_call',
    tags: ['openai', 'retrieval'],
});

export async function externalSearchTool (neededInformation: string) {
    try {
        return await externalSearchRunnable.invoke(neededInformation);
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Failed to search the web: ${errorMessage}. Please try again or rephrase the query.`);
    }
}

export const externalSearchToolForAgent = tool(
    async ({ neededInformation }) => {
        try {
            return await externalSearchRunnable.invoke(neededInformation);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to search the web: ${errorMessage}. Please try again or rephrase the query.`);
        }
    },
    {
        name: GET_EXTERNAL_INFO,
        description:
            `Search the web for external/real-world information related to the novel "${config.BOOK_NAME}". ` +
            `Use this for general/real-world knowledge such as historical context, cultural references, literary analysis, author background, etc.`,
        schema: z.object({
            neededInformation: z.string().describe(
                'A description of the specific external/real-world information needed. ' +
                'Be specific and descriptive.'
            ),
        }),
    }
);
