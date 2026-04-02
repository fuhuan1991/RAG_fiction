import 'dotenv/config';

import config from './config.ts';

/** Input type for answerSingleQuestion */
export interface AnswerSingleQuestionInput {
    /** The user's natural-language question about the novel */
    userQuery: string;
}

/** Returned when the question is deemed irrelevant or impractical */
export interface IrrelevantQuestionResult {
    isRelevant: false;
    /** Brief explanation of why the question was rejected */
    reason: string;
}

/** Returned when the question is relevant and an answer is generated */
export interface RelevantQuestionResult {
    isRelevant: true;
    /** The generated answer (markdown-formatted with Summary & Supporting Evidence) */
    answer: string;
}

/** Discriminated union – the return type of answerSingleQuestion */
export type AnswerSingleQuestionOutput =
    | IrrelevantQuestionResult
    | RelevantQuestionResult;

import { createAgent } from "langchain";
import { relevantCheckPromptTemplate, relevantCheckOutput } from "./promptTemplates/relevantCheckPrompt";
import { searchPromptTemplate, searchOutput } from "./promptTemplates/searchPrompt";
import { answerGenerationPromptTemplate } from "./promptTemplates/answerGenerationPrompt.ts";
import { createInternalInfoTool } from "./tools/internalInfoTool.ts";
import { createExternalInfoTool } from "./tools/externalInfoTool.ts";

export async function answerSingleQuestion({ userQuery }: AnswerSingleQuestionInput): Promise<AnswerSingleQuestionOutput> {

    const [getInternalInfo, acquiredChunks] = createInternalInfoTool();
    const [getExternalInfo, acquiredExternalInfo] = createExternalInfoTool();

    const relevantCheckAgent = createAgent({
        model: config.GPT_MODEL,
        tools: [],
        responseFormat: relevantCheckOutput
    });

    const searchAgent = createAgent({
        model: config.GPT_MODEL,
        tools: [ getInternalInfo, getExternalInfo ],
        responseFormat: searchOutput
    });

    const answerAgent = createAgent({
        model: config.GPT_MODEL,
    });

    const relevantCheckPrompt = await relevantCheckPromptTemplate.formatMessages({ userQuery });
    const searchPrompt = await searchPromptTemplate.formatMessages({ userQuery });
    
    const relevantCheckResult = await relevantCheckAgent.invoke({ messages: relevantCheckPrompt });

    if (!relevantCheckResult.structuredResponse.isRelevant) {
        return {
            isRelevant: false,
            reason: relevantCheckResult.structuredResponse.reason
        }
    }

    const searchResult = await searchAgent.invoke({ messages: searchPrompt });

    const internalContextString = acquiredChunks
        .map((chunk, i) => `<chunk id="${i + 1}">\n${chunk}\n</chunk>`)
        .join("\n");

    const externalContextString = acquiredExternalInfo
        .map((chunk, i) => `<external-info id="${i + 1}">\n${chunk}\n</external-info>`)
        .join("\n");

    const answerGenerationPrompt = await answerGenerationPromptTemplate.formatMessages({ userQuery, internalContext: internalContextString, externalContext: externalContextString});
    
    const finalResult = await answerAgent.invoke({ messages: answerGenerationPrompt });
    const aiMessage = finalResult.messages[finalResult.messages.length - 1];

    return {
        isRelevant: true,
        answer: aiMessage.content as string,
    };
}
