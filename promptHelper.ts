import { ChatPromptTemplate } from '@langchain/core/prompts';


export const singleQueryPrompt = ChatPromptTemplate.fromMessages([
  ['system', 
    'You are a helpful assistant answering user\'s question about a fictional novel. ' + 
    'Answer the question based only on the provided CONTEXT. If the CONTEXT does not contain enough information to answer, you should point it out in the response. ' + 
    'Provide no more than 3 references from the CONTEXT to support your answer if needed, and show some original text from CONTEXT.'],
  ['human', 'CONTEXT:\n {chunkString} \n\n QUESTION: {userQuery}']
]);

