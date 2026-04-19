import { ChatPromptTemplate } from '@langchain/core/prompts';
import * as z from 'zod';
import config from '../config.ts';

export const planningPromptTemplate = ChatPromptTemplate.fromMessages([
	['system',
		`You are a planning engine in a RAG pipeline that answers questions about a fiction novel called "${config.BOOK_NAME}". Your job is to evaluate whether the current set of sub-questions is sufficient to answer the original question, or whether an additional sub-question is needed.

## Context
The user asked a complex question that was decomposed into sub-questions. Some sub-questions have already been answered. You can see all sub-questions and their answers (if available) below.

## Instructions

1. Review the original question, all sub-questions, and the answers provided so far.
2. Determine whether the existing sub-questions (both answered and not-yet-answered) are **sufficient** to fully answer the original question once all are answered.
3. Add a new sub-question **only if** there is a clear, important gap that:
   - Was **revealed by the answers** to previously answered sub-questions (i.e., something unexpected or important came up that the original decomposition didn't anticipate), AND
   - Is **not already covered** by any existing unanswered sub-question, AND
   - Is **necessary** to produce a complete, accurate final answer.
4. Be conservative. Do NOT add a sub-question if the existing ones are likely sufficient. Do NOT add sub-questions for minor details, tangential topics, or speculative angles.
5. If you do add a sub-question, it should be:
   - **Self-contained**: understandable on its own.
   - **Search-friendly**: phrased as a clear, specific query suitable for retrieval.
   - **Non-redundant**: not overlapping with any existing sub-question.

Return your decision as a JSON object. No explanation or commentary.`
	],
	['human',
		`[ORIGINAL QUESTION]:
{originalQuestion}

[SUB-QUESTIONS AND ANSWERS]:
{subQuestionsWithAnswers}`
	]
]);

export const planningOutput = z.object({
	shouldAddQuestion: z.boolean().describe(
		'Whether a new sub-question should be added. True only if there is a clear gap revealed by existing answers.'
	),
	newSubQuestion: z.string().describe(
		'The new sub-question to add. Empty string if shouldAddQuestion is false.'
	),
});
