import { Pinecone } from '@pinecone-database/pinecone'
import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';

export const handler = async (event, context) => {
    try {
        const record = event.Records[0];
        const bodyString = record.body;
        const body = JSON.parse(bodyString);
        const chunkId = body.chunkId;
        const chunkIndex = body.chunkIndex;
        const chunkText = body.chunkText;

        console.log('With context:', JSON.stringify(context));
        console.log("----------------Fetched a chunk from SQS queue, chunkId: " + chunkId);

        const secretName = process.env.PINECONE_SECRET_NAME;
        const region = "us-east-1";
        const client = new SecretsManagerClient({
            region: region
        });

        const response = await client.send(
            new GetSecretValueCommand({
                SecretId: secretName,
                VersionStage: "AWSCURRENT"
            })
        );
        
        const secretObject = JSON.parse(response.SecretString);
        const PINECONE_API_KEY = secretObject.PINECONE_API_KEY;
        const PINECONE_HYBRID_INDEX_NAME = secretObject.PINECONE_HYBRID_INDEX_NAME;
        const PINECONE_INDEX_NAMESPACE = secretObject.PINECONE_INDEX_NAMESPACE;

        const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
        const index = pc.index(PINECONE_HYBRID_INDEX_NAME).namespace(PINECONE_INDEX_NAMESPACE);

        const denseEmbedding = await pc.inference.embed({
            model: 'llama-text-embed-v2',
            inputs: [ chunkText ],
            parameters: {
              inputType: 'passage',
              truncate: 'END',
            }
        });

        const sparseEmbedding = await pc.inference.embed({
            model: 'pinecone-sparse-english-v0',
            inputs: [ chunkText ],
            parameters: {
              inputType: 'passage',
              truncate: 'END',
            }
        });

        await index.upsert({
            namespace: PINECONE_INDEX_NAMESPACE,
            records: [{
                id: chunkId,
                metadata: {
                    book_title: body.bookTitle,
                    chunk_text: chunkText
                },
                values: denseEmbedding.data[0].values,
                sparseValues: {
                    indices: sparseEmbedding.data[0].sparseIndices,
                    values: sparseEmbedding.data[0].sparseValues
                },
            }]
        })

        console.log("----------------A chunk is uploaded to vector database, chunkId: " + chunkId);

        return {
            statusCode: 200,
            body: JSON.stringify({
                message: "A chunk is uploaded to vector database, chunkId: " + chunkId,
            })
        };
    } catch (error) {
        console.error('Error occurred:', {
            errorMessage: error instanceof Error ? error.message : 'Unknown error',
            errorStack: error instanceof Error ? error.stack : undefined,
            requestId: context.requestId,
            functionName: context.functionName,
            event: JSON.stringify(event)
        });

        return {
            statusCode: 500,
            body: JSON.stringify({
                message: 'Internal server error',
                requestId: context.requestId
            })
        };
    }
};

