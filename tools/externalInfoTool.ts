import * as z from 'zod';
import { tool } from 'langchain';
import OpenAI from 'openai';
import config from '../config.ts';

export const GET_EXTERNAL_INFO: string = 'get_external_info';

const openai = new OpenAI({ apiKey: config.OPENAI_API_KEY });

/**
 * Perform a web search using OpenAI's Responses API with web_search_preview.
 */
async function externalSearch(query: string): Promise<string> {

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
}

/**
 * Factory function that creates a new getExternalInfo tool instance.
 * Each instance maintains its own list of acquired external information
 * to track results across multiple calls within the same session.
 *
 * Usage: Call this once per user request/session so that acquired info
 * state is not shared across different users.
 */
export function createExternalInfoTool() {
    // Per-session list to store external info retrieved during this session
    const acquiredExternalInfo: string[] = [];

    const newTool = tool(
        async ({ neededInformation }) => {

            console.log('----LLM wants to know this from internet: ' + neededInformation);

            try {
                const result = await externalSearch(neededInformation);
                acquiredExternalInfo.push(result);
                return result;
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                console.error('----externalSearch failed:', errorMessage);
                return `Failed to search the web: ${errorMessage}. Please try again or rephrase the query.`;
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

    return [newTool, acquiredExternalInfo] as const;
}
