import {
    StateGraph,
    START,
    END,
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import * as z from 'zod';
import 'dotenv/config';
import config from './config.ts';
import { createAgent, toolCallLimitMiddleware } from "langchain";

import { relevantCheckPromptTemplate, relevantCheckOutput } from "./promptTemplates/relevantCheckPrompt";
import { decompositionPromptTemplate, decompositionOutput } from './promptTemplates/questionDecompositionPrompt.ts';
import { sourceClassifierPromptTemplate, sourceClassifierOutput } from './promptTemplates/sourceClassifierPrompt.ts';
import { answerSubQuestionPromptTemplate, answerSubQuestionOutput } from './promptTemplates/answerSubQuestionPrompt.ts';
import { answerFinalQuestionPromptTemplate, answerFinalQuestionOutput } from './promptTemplates/answerFinalQuestionPrompt.ts';
import { planningPromptTemplate, planningOutput } from './promptTemplates/planningPrompt.ts';
import { internalSearchTool } from './tools/internalInfoTool.ts';
import { externalSearchTool, externalSearchToolForAgent } from './tools/externalInfoTool.ts';
import { ChunkResult } from './pineconeHandler.ts';

export const QuestionStateDefinition = z.object({
    originalQuestion: z.string().default(''),
    isRelevant: z.boolean().default(false),
    questionStack: z.array(
        z.object({
            question: z.string(),
            answer: z.string(),
        })
    ).default([]),
    decomposed: z.boolean().default(false),
    questionIndex: z.number().default(-1),
    chunks: z.array(
        z.object({
            id: z.string(),
            text: z.string(),
        })
    ).default([]),
    seenChunkIds: z.set(z.string()).default(new Set()),
    externalInfo: z.array(z.string()).default([]),
    finalAnswer: z.string().default(''),
});

export type QuestionAgentState = z.infer<typeof QuestionStateDefinition>;

const QUESTION_DECOMPOSITION_TEMPERATURE = 0.1;

const llm = new ChatOpenAI({
    model: config.GPT_MODEL,
    temperature: QUESTION_DECOMPOSITION_TEMPERATURE,
    apiKey: config.OPENAI_API_KEY,
});

async function relevantCheck(state: QuestionAgentState) {
    const originalQuesiton = state.originalQuestion;
    const relevantCheckPrompt = await relevantCheckPromptTemplate.formatMessages({ userQuery: originalQuesiton });
    const relevantCheckAgent = createAgent({
        name: "RelevantCheckAgent",
        model: config.GPT_MODEL,
        tools: [ externalSearchToolForAgent ],
        responseFormat: relevantCheckOutput,
        middleware: [
            toolCallLimitMiddleware({ runLimit: config.RELEVANT_CHECK_MAX_TOOL_CALLS }),
        ],
    });

    const relevantCheckResult = await relevantCheckAgent.invoke({ messages: relevantCheckPrompt });

    if (!!relevantCheckResult.structuredResponse.isRelevant) {
        return {
            isRelevant: true,
        }
    } else {
        return {
            isRelevant: false,
            finalAnswer: relevantCheckResult.structuredResponse.reason,
        }
    }
}

function routerAfterRelevantCheck(state: QuestionAgentState): string {
    if (!! state.isRelevant) {
        return 'decomposition_node';
    } else {
        return END;
    }
}

async function decomposition(state: QuestionAgentState) {
    const structuredLlm = llm.withStructuredOutput(decompositionOutput);
    const originalQuesiton = state.originalQuestion;
    const decompositionPrompt = await decompositionPromptTemplate.formatMessages({ userQuery: originalQuesiton });
    const { decomposed, subQueries } = await structuredLlm.invoke(decompositionPrompt, { runName: "decomposition_llm_call" });

    if (!!decomposed) {
        const questionStack = [];
        for (const subQuery of subQueries) {
            const item = {
                question: subQuery,
                answer: '',
            };
            questionStack.unshift(item);
        }

        return { 
            decomposed: true, 
            questionIndex: questionStack.length - 1, 
            questionStack 
        };
    } else {
        return { 
            decomposed: false, 
            questionIndex: -1,
        }; 
    }  
}

async function planning(state: QuestionAgentState) {
    const { decomposed, questionIndex, questionStack, originalQuestion } = state;

    // Skip planning if:
    // - Question was not decomposed (simple question)
    // - We're in the final answer phase (questionIndex === -1)
    // - No sub-questions have been answered yet (nothing to learn from)
    // - Already at the sub-question cap
    const hasAnsweredSubQuestions = questionStack.some(q => q.answer !== '');
    if (!decomposed 
        || questionIndex === -1 
        || !hasAnsweredSubQuestions 
        || questionStack.length >= config.MAX_SUB_QUESTIONS) {
        return {};
    }

    // Build a formatted string showing all sub-questions with their answer status
    // Processing order is high index → low index, so we display from high to low
    let subQuestionsWithAnswers = '';
    let displayIndex = 1;
    for (let i = questionStack.length - 1; i >= 0; i--) {
        const { question, answer } = questionStack[i];
        const status = answer !== '' ? 'answered' : 'not yet answered';
        subQuestionsWithAnswers += `Sub-question ${displayIndex} (${status}): ${question}\n`;
        if (answer !== '') {
            subQuestionsWithAnswers += `Answer ${displayIndex}: ${answer}\n`;
        }
        subQuestionsWithAnswers += '\n';
        displayIndex++;
    }

    const structuredLlm = llm.withStructuredOutput(planningOutput);
    const planningPrompt = await planningPromptTemplate.formatMessages({
        originalQuestion,
        subQuestionsWithAnswers,
    });
    const { shouldAddQuestion, newSubQuestion } = await structuredLlm.invoke(planningPrompt, { runName: "planning_llm_call" });

    if (shouldAddQuestion && newSubQuestion) {
        const updatedStack = structuredClone(questionStack);
        // Insert the new sub-question at questionIndex + 1 so it gets processed next
        // (processing goes from high index → low index)
        const insertIndex = questionIndex + 1;
        updatedStack.splice(insertIndex, 0, { question: newSubQuestion, answer: '' });
        return {
            questionStack: updatedStack,
            questionIndex: questionIndex + 1,
        };
    }

    return {};
}

async function gatherInformation(state: QuestionAgentState) {
    const questionIndex = state.questionIndex;
    let questionToSearch;

    if (questionIndex === -1) {
        // gather information for original question
        if (!!state.decomposed) {
            // Complex question. No need to retrieve information, it can be answered with the existing information retrieved for sub-questions
            return {};
        } else {
            // Simple question. Need to retrieve information, since there is no existing information
            questionToSearch = state.originalQuestion;
        }
    } else {
        // gather information for sub-questions
        questionToSearch = state.questionStack[questionIndex].question;
    }

    const structuredLlm = llm.withStructuredOutput(sourceClassifierOutput);
    const sourceClassifierPrompt = await sourceClassifierPromptTemplate.formatMessages({ userQuery: questionToSearch });
    const { needInternalSearch, needExternalSearch } = await structuredLlm.invoke(sourceClassifierPrompt, { runName: "source_classification_llm_call" });

    const seenChunkIds = new Set(state.seenChunkIds);
    let chunks = state.chunks;
    const externalInfo = state.externalInfo;

    if (needInternalSearch) {
        // seenChunkIds will be updated in internalSearchTool()
        const newChunks = await internalSearchTool(questionToSearch, seenChunkIds);
        chunks = chunks.concat(newChunks);
    }

    if (needExternalSearch) {
        const externalSearchResult = await externalSearchTool(questionToSearch);
        externalInfo.push(externalSearchResult);
    }

    return { chunks, seenChunkIds, externalInfo };
}

async function answering(state: QuestionAgentState) {
    const questionIndex = state.questionIndex;
    const externalInfo: string[] = state.externalInfo;
    const chunks: ChunkResult[] = state.chunks;
    const questionStack = structuredClone(state.questionStack);

    let previousQAString = '';
    let index = 1;
    for (let i = questionStack.length - 1; i > questionIndex; i --) {
        const { question, answer } = questionStack[i];
        previousQAString += `Question ${index}: ` + question + '\n' + `Answer ${index}: ` + answer + '\n';
        index++
    }

    const externalInfoString = externalInfo.map(
        (item, i) => `External info ${i + 1}: ${item}`
    ).join("\n");

    const chunksString = chunks.map(
        (item, i) => `Book chunk ${i + 1}: ${item.text}`
    ).join("\n");

    if (questionIndex === -1) {
        // Answer final question
        const questionToAnswer = state.originalQuestion;
        const answerFinalQuestionPrompt = await answerFinalQuestionPromptTemplate.formatMessages({ 
            previousQA: previousQAString,
            internalContext: chunksString,
            externalContext: externalInfoString,
            finalQuestion: questionToAnswer
        });

        const structuredLlm = llm.withStructuredOutput(answerFinalQuestionOutput);
        const { finalQuestionAnswer } = await structuredLlm.invoke(answerFinalQuestionPrompt, { runName: "answer_final-question_llm_call" });
        return { finalAnswer: finalQuestionAnswer };
    } else {
        // Answer sub-question
        const questionToAnswer = questionStack[questionIndex].question;
        const answerSubQuestionPrompt = await answerSubQuestionPromptTemplate.formatMessages({ 
            previousQA: previousQAString,
            internalContext: chunksString,
            externalContext: externalInfoString,
            subQuestion: questionToAnswer
        });

        const structuredLlm = llm.withStructuredOutput(answerSubQuestionOutput);
        const { subQuestionAnswer } = await structuredLlm.invoke(answerSubQuestionPrompt, { runName: "answer_sub-question_llm_call" });
        const newSubQuestionItem = { question: questionStack[questionIndex].question, answer: subQuestionAnswer };
        
        questionStack[questionIndex] = newSubQuestionItem;
        return { questionStack, questionIndex: questionIndex - 1 };
    }
}

function routerAfterAnswer(state: QuestionAgentState): string {
    if (state.finalAnswer === '') {
        return 'planning_node';
    } else {
        return END;
    }
}

export const graph = new StateGraph(QuestionStateDefinition)
    // Nodes:
    .addNode('relevant_check_node', relevantCheck)
    .addNode('decomposition_node', decomposition)
    .addNode('planning_node', planning)
    .addNode('gather_information_node', gatherInformation)
    .addNode('answering_node', answering)
    // Edges:
    .addEdge(START, 'relevant_check_node')
    .addConditionalEdges('relevant_check_node', routerAfterRelevantCheck, ['decomposition_node', END])
    .addEdge('decomposition_node', 'planning_node')
    .addEdge('planning_node', 'gather_information_node')
    .addEdge('gather_information_node', 'answering_node')
    .addConditionalEdges('answering_node', routerAfterAnswer, ['planning_node', END])
    .compile();
