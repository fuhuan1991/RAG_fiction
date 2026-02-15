import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { SQSClient, SendMessageCommand } from "@aws-sdk/client-sqs";
import readline from "readline";

const s3 = new S3Client({});
const sqs = new SQSClient({ region: "us-east-1" });
const MAX_TOKENS = 700;
const OVERLAP_TOKENS = 120;

export const handler = async (event, context) => {
    try {
        console.log('Processing event:', JSON.stringify(event));
        console.log('With context:', JSON.stringify(context));
        console.log("----------------Start chunking: " + event.Records[0].s3.object.key);
        const bucket = event.Records[0].s3.bucket.name;
        const s3FileName = event.Records[0].s3.object.key;
        const bookTitle = s3FileName.split('.')[0]; // remove '.txt'

        const chunksEmitted = await chunk(bucket, s3FileName, bookTitle);

        console.log(`----------------Chunking completed - total chunks emitted: ${chunksEmitted}`);

        return {
            statusCode: 200,
            body: JSON.stringify({
                message: `Chunking completed - total chunks emitted: ${chunksEmitted}`
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

        // Return error response
        return {
            statusCode: 500,
            body: JSON.stringify({
                message: 'Internal server error',
                requestId: context.requestId
            })
        };
    }
};

async function chunk(bucket, s3FileName, bookTitle) {
    const command = new GetObjectCommand({ Bucket: bucket, Key: s3FileName });
    const response = await s3.send(command);

    const rl = readline.createInterface({
        input: response.Body,
        crlfDelay: Infinity
    });

    let paragraphs = [];
    let currentChunk = [];
    let currentTokens = 0;
    let chunkIndex = 0;
    let chunksEmitted = 0;

    for await (const line of rl) {
        if (line.trim() === "") {
            const paragraphText = paragraphs.join(" ");
            paragraphs = [];

            const tokens = estimateTokens(paragraphText);

            if (currentTokens + tokens > MAX_TOKENS) {
                await emitChunk(currentChunk, chunkIndex++, bookTitle);
                chunksEmitted++;
                currentChunk = getOverlap(currentChunk);
                currentTokens = estimateTokens(currentChunk.join("\n"));
            }

            currentChunk.push(paragraphText);
            currentTokens += tokens;
        } else {
            paragraphs.push(line);
        }
    }

    // Flush remaining content
    if (currentChunk.length > 0) {
        await emitChunk(currentChunk, chunkIndex++, bookTitle);
        chunksEmitted++;
    }

    return chunksEmitted;


}

async function emitChunk(paragraphs, index, bookTitle) {
    const text = paragraphs.join("  ");

    console.log(JSON.stringify({
        chunkId: index,
        text
    }));

    const params = {
        QueueUrl: process.env.QUEUE_URL,
        MessageBody: JSON.stringify({
            chunkId: bookTitle + "_" + index,
            chunkIndex: index,
            chunckText: text,
            bookTitle: bookTitle,
            timestamp: Date.now(),
            language: "en"
        }),
    };

    const command = new SendMessageCommand(params);
    const response = await sqs.send(command);

    console.log("Chunk_" + index + " sent to SQS", response.MessageId);
}

function getOverlap(paragraphs) {
    let tokens = 0;
    const overlap = [];

    for (let i = paragraphs.length - 1; i >= 0; i--) {
        tokens += estimateTokens(paragraphs[i]);
        overlap.unshift(paragraphs[i]);
        if (tokens >= OVERLAP_TOKENS) break;
    }

    return overlap;
}

function estimateTokens(text) {
    // Rough but effective for English
    return Math.ceil(text.split(/\s+/).length / 0.75);
}