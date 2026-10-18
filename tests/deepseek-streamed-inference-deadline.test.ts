import { test } from "bun:test";
import assert from "node:assert/strict";

import { BUFFERED_INFERENCE_DEADLINE_MS, setBufferedInferenceDeadlineMsForTest } from "../src/inference-deadline.ts";
import { handleChatCompletions } from "../src/chat/envelope.ts";
import { handleResponses } from "../src/responses-handler.ts";
import { withFetchMock } from "./helpers/openai-compat-harness.ts";

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

test("streamed chat completion outlives BUFFERED_INFERENCE_DEADLINE_MS while delivering events", async () => {
  const originalKey = process.env.DEEPSEEK_API_KEY;
  process.env.DEEPSEEK_API_KEY = "test-deepseek-key";

  // Set the buffered deadline to a short window (50ms) for testing
  setBufferedInferenceDeadlineMsForTest(50);

  try {
    const encoder = new TextEncoder();

    // Stream delivers events at 0ms, 30ms, 60ms, and 90ms (total ~90ms, outliving 50ms)
    const streamHandler = () => {
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          const chunk1 = {
            id: "chatcmpl-stream-1",
            object: "chat.completion.chunk",
            created: 1_780_000_000,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: { role: "assistant", content: "part1" }, finish_reason: null }],
          };
          const chunk2 = {
            id: "chatcmpl-stream-1",
            object: "chat.completion.chunk",
            created: 1_780_000_001,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: { content: "part2" }, finish_reason: null }],
          };
          const chunk3 = {
            id: "chatcmpl-stream-1",
            object: "chat.completion.chunk",
            created: 1_780_000_002,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          };

          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk1)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk2)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk3)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        },
      });

      return new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const response = await withFetchMock(streamHandler, () =>
      handleChatCompletions(
        new Request("https://ai.ubq.fi/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "deepseek-flash",
            messages: [{ role: "user", content: "hi" }],
            stream: true,
          }),
        })
      )
    );

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "text/event-stream");

    // Read all streamed frames to verify the stream completed without deadline abort
    const text = await response.text();
    assert.ok(text.includes("part1"), "stream must contain part1");
    assert.ok(text.includes("part2"), "stream must contain part2");
    assert.ok(text.includes("[DONE]"), "stream must finish with [DONE]");
  } finally {
    setBufferedInferenceDeadlineMsForTest(null);
    if (originalKey === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = originalKey;
  }
});

test("streamed responses outlives BUFFERED_INFERENCE_DEADLINE_MS while delivering events", async () => {
  const originalKey = process.env.DEEPSEEK_API_KEY;
  process.env.DEEPSEEK_API_KEY = "test-deepseek-key";

  setBufferedInferenceDeadlineMsForTest(50);

  try {
    const encoder = new TextEncoder();

    const streamHandler = () => {
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          const chunk1 = {
            id: "chatcmpl-stream-2",
            object: "chat.completion.chunk",
            created: 1_780_000_000,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: { role: "assistant", content: "hello" }, finish_reason: null }],
          };
          const chunk2 = {
            id: "chatcmpl-stream-2",
            object: "chat.completion.chunk",
            created: 1_780_000_001,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: { content: " world" }, finish_reason: null }],
          };
          const chunk3 = {
            id: "chatcmpl-stream-2",
            object: "chat.completion.chunk",
            created: 1_780_000_002,
            model: "deepseek-flash",
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          };

          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk1)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk2)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk3)}\n\n`));
          await sleep(30);
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        },
      });

      return new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const response = await withFetchMock(streamHandler, () =>
      handleResponses(
        new Request("https://ai.ubq.fi/v1/responses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "deepseek-flash",
            input: "hi",
            stream: true,
          }),
        })
      )
    );

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "text/event-stream");

    const text = await response.text();
    assert.ok(text.includes("response.output_item.added") || text.includes("response.completed"), "stream must contain responses events");
    assert.ok(text.includes("[DONE]") || text.includes("response.completed"));
  } finally {
    setBufferedInferenceDeadlineMsForTest(null);
    if (originalKey === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = originalKey;
  }
});
