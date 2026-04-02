import * as readline from "readline";
import { answerSingleQuestion } from "./answerSingleQuestion.ts";
import config from './config.ts';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.question(`Please ask a question about the book ${config.BOOK_NAME}: `, async (userQuery) => {
  rl.close();

  try {
    const result = await answerSingleQuestion({ userQuery });

    if (result.isRelevant) {
      console.log(result.answer);
    } else {
      console.log("Question deemed irrelevant:");
      console.log(result.reason);
    }
  } catch (error) {
    console.error("Error:", error);
  }
});
