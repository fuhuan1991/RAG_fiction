# RAG Fiction — Question Answering for *Babel*

A Retrieval-Augmented Generation (RAG) system that answers questions about the novel **Babel** by R.F. Kuang. It combines intelligent question decomposition, hybrid vector search, and LLM-powered answer synthesis to handle both simple factual queries and complex, multi-faceted questions about the book.

---

## How It Works

The system is built as a stateful workflow using **LangGraph**. Each incoming question travels through five nodes:

```
START
  └─► relevant_check_node      — Is this question about Babel?
        └─► decomposition_node  — Simple question or complex (needs sub-questions)?
              └─► planning_node — Are there information gaps? Add sub-questions if needed.
                    └─► gather_information_node — Search book (Pinecone) and/or web.
                          └─► answering_node    — Answer current sub-question or synthesize final answer.
                                └─► [router]    — More sub-questions? → planning_node. Done? → END.
```

### Node Descriptions

| Node | Role |
|------|------|
| `relevant_check_node` | Validates the question is about the novel. Routes to END if not relevant. |
| `decomposition_node` | Decides if the question needs sub-questions. Generates up to 3 ordered sub-questions for complex queries. |
| `planning_node` | Reviews answers so far and dynamically adds more sub-questions if information gaps are detected. |
| `gather_information_node` | Runs hybrid vector search in Pinecone and/or web search depending on what the question needs. |
| `answering_node` | Answers the current sub-question using all prior Q&A context, or synthesizes the final answer once all sub-questions are resolved. |

---

## Key Features

- **Hybrid Search** — Combines dense (semantic) and sparse (keyword) embeddings in Pinecone. Weighted by an `alpha` parameter (default 0.7 = 70% semantic, 30% keyword).
- **Multi-Query Expansion** — Each retrieval generates N alternative phrasings of the query and merges results to improve recall.
- **Dynamic Planning** — The planning node re-evaluates after each sub-answer and can inject new sub-questions when gaps are discovered.
- **Deduplication** — A `seenChunkIds` set tracks retrieved chunks across the whole session to avoid redundant context.
- **Structured Outputs** — Every LLM call returns validated JSON via Zod schemas, ensuring deterministic behavior.
- **LangSmith Tracing** — All major operations are tagged for debugging and monitoring.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraphjs) | Stateful workflow orchestration |
| [LangChain](https://js.langchain.com/) | LLM chains, prompt templates, runnable abstractions |
| [Pinecone](https://www.pinecone.io/) | Vector database for hybrid search |
| [OpenAI](https://platform.openai.com/) | GPT-4.1-mini (main LLM), GPT-4o-nano (query expansion), web search |
| [Zod](https://zod.dev/) | Schema validation for structured LLM outputs |
| [TypeScript](https://www.typescriptlang.org/) | Language (compiled to ES2022) |
| [Vitest](https://vitest.dev/) | Unit testing |
| [LangSmith](https://smith.langchain.com/) | Observability and tracing |

---

## Project Structure

```
RAG_fiction/
├── src/
│   ├── questionFlow.ts          # Core LangGraph graph — all 5 nodes, routers, state schema
│   ├── config.ts                # Centralized config (API keys, model names, search params)
│   ├── pineconeHandler.ts       # Hybrid vector search engine with retry logic
│   ├── app.ts                   # Entry point
│   ├── tools/
│   │   ├── internalInfoTool.ts  # Query expansion + Pinecone search
│   │   └── externalInfoTool.ts  # Web search via OpenAI web_search_preview
│   └── promptTemplates/         # 8 Zod-typed prompt templates (one per LLM decision point)
└── tests/
    └── pineconeHandler.test.ts  # 30+ unit tests for hybrid search logic
```

---

## Setup

### Prerequisites

- Node.js 18+
- A Pinecone index pre-populated with embeddings from *Babel*
- API keys for OpenAI, Pinecone, and LangSmith

### Install

```bash
npm install
```

### Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG_fiction
```

---

## Usage

```bash
npm start          # Run the app (app.ts)
npm run cli_test   # Run CLI test
npm run typecheck  # TypeScript type checking
npm test           # Run Vitest unit tests
```

---

## State Schema

The LangGraph state is validated with Zod and carries the full conversation context through the graph:

```typescript
{
  originalQuestion: string,
  isRelevant: boolean,
  decomposed: boolean,
  questionStack: { question: string, answer: string }[],
  questionIndex: number,   // -1 signals the final answer phase
  chunks: { id: string, text: string }[],
  seenChunkIds: Set<string>,
  externalInfo: string[],
  finalAnswer: string
}
```
