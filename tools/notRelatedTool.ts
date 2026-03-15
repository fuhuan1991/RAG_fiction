import * as z from 'zod';
import { tool } from 'langchain';
import config from '../config.ts';

export const NOT_RELATED: string = 'not_related';

export const notRelated = tool(
    async ({ reply }) => {
        console.log('----notRelated:', reply);
        return reply;
    },
    {
        name: NOT_RELATED,
        description:
            `Use this tool ONLY when you are certain that the user's question is NOT related to the novel "${config.BOOK_NAME}". ` +
            `If there is any possibility the question could be related to the novel, do NOT use this tool. ` +
            `When called, provide a short explanation of why the question is not related to the novel.`,
        schema: z.object({
            reply: z.string().describe(
                'A short, polite explanation to the user about why their question is not related to the novel.'
            )
        }),
    }
);
