import { ChatPromptTemplate } from '@langchain/core/prompts';
import * as z from 'zod';
import config from '../config.ts';

export const queryExpansionPromptTemplate = ChatPromptTemplate.fromMessages([
	['system',
		`You are a query expansion assistant for a vector search system that retrieves passages from the novel "${config.BOOK_NAME}".

You will receive a single, focused search query. Your job is to generate {alternativeCount} alternative phrasings of the **same query intent** to improve vector search recall.

Rules:
- Each alternative must target the **same information need** as the original query — do NOT change the topic or broaden the scope.
- Vary the phrasing, word choice, and sentence structure so that each alternative captures a different lexical/semantic angle.
- Use character names, place names, and specific details from the novel when possible.
- Write each alternative as a natural-language statement (not keywords or bullet points), matching the style of the novel's prose.
- Do NOT include meta-language like "Find passages about..." or "Search for...". Directly state the content.
- Each alternative should be roughly the same length as the original query (1–2 sentences).

Respond with a JSON object containing exactly one field:
- "alternatives": an array of exactly {alternativeCount} strings, each being an alternative phrasing of the original query.

Example:
Original query: "Robin's feelings of guilt about his privilege at Oxford"
Output:
{{"alternatives": ["Robin's inner conflict about benefiting from the colonial system while studying at Babel", "Robin struggling with the moral weight of his advantaged position as a translator at Oxford"]}}

Do NOT include the original query in the alternatives array — only new phrasings.`
	],
	['human', '{originalQuery}']
]);

export const queryExpansionOutput = z.object({
	alternatives: z.array(z.string()).describe(
		'Alternative phrasings of the original query, each targeting the same information need but using different wording.'
	),
});
