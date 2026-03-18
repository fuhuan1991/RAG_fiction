import { ChatPromptTemplate } from '@langchain/core/prompts';
import * as z from 'zod';
import config from '../config.ts';

export const relevantCheckPrompt = ChatPromptTemplate.fromMessages([
	['system',
		`You are a helpful assistant that analyzes whether a user's question is related to the novel "${config.BOOK_NAME}".

Guidelines for determining relevance:
- "Related": The question asks about the novel's plot, characters, setting, themes, writing style, author intent, literary devices, historical/cultural background, or anything that would require knowledge of the novel to answer properly, AND can be reasonably answered in a short response.
- "Not related": The question has absolutely nothing to do with the novel (e.g., asking about the weather, coding help, math problems, other unrelated books or topics).
- "Impractical": The question is related to the novel but requests a task that is too large, complex, or impractical to complete in a short conversation. Examples include but are not limited to: translating the entire book or large portions of it, rewriting the book, listing every character with full descriptions, reproducing the full text, or any request that would essentially require processing the entire novel. These should be treated as NOT relevant.
- When in doubt, treat the question as related — UNLESS it clearly requires processing or producing an unreasonably large amount of content.

You must respond with a JSON object containing exactly these two fields:
- "isRelevant": a boolean value. true if the question is related to the novel AND can be reasonably answered in a short response. false if the question is unrelated to the novel OR is too complex/impractical to complete.
- "reason": a brief string explaining why the question is or is not relevant, or why it is impractical.

Example output for a related question:
{{"isRelevant": true, "reason": "The question asks about a character in the novel."}}

Example output for an unrelated question:
{{"isRelevant": false, "reason": "The question is about a math problem and has nothing to do with the novel."}}

Example output for an impractical question:
{{"isRelevant": false, "reason": "The request to translate the entire book is too large and impractical to complete in a short conversation."}}

Respond ONLY with the JSON object. Do not include any additional text, explanation, or markdown formatting.`
	],
	['human', '{userQuery}']
]);

export const relevantCheckOutput = z.object({
	isRelevant: z.boolean(),
	reason: z.string()
});
