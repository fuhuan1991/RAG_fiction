import 'dotenv/config';

/**
 * Configuration file for pintcone_test
 */

interface ServerConfig {
  port: number | string;
  host: string;
}

interface ApiConfig {
  timeout: number;
  retries: number;
}

interface Config {
  env: string;
  PINECONE_API_KEY: string;
  OPENAI_API_KEY: string;
  INDEX_NAME: string;
  NAME_SPACE: string;
  DENSE_EMBED_MODEL: string;
  SPARSE_EMBED_MODEL: string;
  server: ServerConfig;
  api: ApiConfig;
  TOP_K: number;
  HYBRID_SEARCH_ALPHA: number;
  GPT_MODEL: string;
  BOOK_NAME: string;
  RELEVANT_CHECK_MAX_TOOL_CALLS: number;
  SEARCH_MAX_TOOL_CALLS: number;
  MULTI_QUERY_COUNT: number;
  QUERY_EXPANSION_MODEL: string;
  MAX_SUB_QUESTIONS: number;
}

const config: Config = {
  // Environment
  env: process.env.NODE_ENV || 'development',

  GPT_MODEL: 'gpt-4.1-mini',
  OPENAI_API_KEY: process.env.OPENAI_API_KEY || '',

  PINECONE_API_KEY: process.env.PINECONE_API_KEY || '',
  INDEX_NAME: 'test-hybrid-index-1',
  NAME_SPACE: '__default__',
  DENSE_EMBED_MODEL: 'llama-text-embed-v2',
  SPARSE_EMBED_MODEL: 'pinecone-sparse-english-v0',
  TOP_K: 10,
  // Weight between dense (semantic) and sparse (keyword) search.
  // 1.0 = pure semantic, 0.0 = pure keyword, 0.5 = equal weight
  HYBRID_SEARCH_ALPHA: 0.7,
  BOOK_NAME: 'Babel',

  // Max tool calls (per run) for each agent
  RELEVANT_CHECK_MAX_TOOL_CALLS: 3,
  SEARCH_MAX_TOOL_CALLS: 10,

  // Multi-query expansion: total number of queries (original + alternatives)
  MULTI_QUERY_COUNT: 3,
  // A fast, cheap model for generating alternative query phrasings
  QUERY_EXPANSION_MODEL: 'gpt-4.1-nano',

  // Max total sub-questions allowed (original decomposed + dynamically added)
  MAX_SUB_QUESTIONS: 5,

  // Server Configuration
  server: {
    port: process.env.PORT || 3000,
    host: process.env.HOST || 'localhost',
  },

  // API Configuration
  api: {
    timeout: 30000,
    retries: 3,
  },
};

export default config;