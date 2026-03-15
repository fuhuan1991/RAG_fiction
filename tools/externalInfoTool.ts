import * as z from 'zod';
import { tool } from 'langchain';
import OpenAI from 'openai';
import config from '../config.ts';

export const GET_EXTERNAL_INFO: string = 'external_info';

/**
 * Perform a web search using OpenAI's Responses API with web_search_preview.
 */
async function externalSearch(query: string): Promise<string> {
    const openai = new OpenAI({ apiKey: config.OPENAI_API_KEY });

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

export const getExternalInfo = tool(
    async ({ infoRequests }) => {

        const results: string[] = [];

        console.log('The following external information is needed:');
        for (const item of infoRequests) {
            console.log('-' + item.neededInformation);
        }

        await Promise.all(
            infoRequests.map(async (request) => {
                const { neededInformation } = request;
                const externalResult = await externalSearch(neededInformation);
                results.push(externalResult);
            })
        );

        return results;
    },
    {
        name: GET_EXTERNAL_INFO,
        description:
            `Search the web for external/real-world information related to the novel "${config.BOOK_NAME}". ` +
            `This tool accepts multiple information requests at once, each specifying what information is needed. ` +
            `Use this for general/real-world knowledge such as historical context, cultural references, literary analysis, author background, etc. ` +
            `Returns an array of strings, one result per request.`,
        schema: z.object({
            infoRequests: z.array(
                z.object({
                    neededInformation: z.string().describe(
                        'A description of the specific external/real-world information needed. ' +
                        'Be specific and descriptive.'
                    ),
                })
            ).describe(
                'An array of information requests. Each request specifies what external information is needed.'
            )
        }),
    }
);
