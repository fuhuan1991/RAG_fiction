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
  GPT_MODEL: string;
  BOOK_NAME: string;
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
  BOOK_NAME: 'Babel',

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