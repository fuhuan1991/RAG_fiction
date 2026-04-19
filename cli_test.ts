import * as readline from "readline";
import { graph } from "./questionFlow.ts";
import config from "./config.ts";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.question(`Please ask a question about the book ${config.BOOK_NAME}: `, async (userQuery) => {
  rl.close();

  try {
    const result = await graph.invoke({ originalQuestion: userQuery });

    if (result.finalAnswer) {
      console.log("\n--- Answer ---");
      console.log(result.finalAnswer);
    } else {
      console.log("No answer was generated.");
    }
  } catch (error) {
    console.error("Error:", error);
  }
});
