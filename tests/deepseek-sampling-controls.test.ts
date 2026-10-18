import { test } from "bun:test";
import assert from "node:assert/strict";

import { toDeepSeekResponsesChatBody } from "../src/deepseek/chat-projection.ts";
import { DEEPSEEK_RESPONSES_PROFILE, LITHOS_RESPONSES_PROFILE } from "../src/deepseek/responses.ts";

test("deepseek responses: forwards temperature and top_p when reasoning effort is none", () => {
  const result = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      temperature: 0,
      top_p: 0.85,
    },
    "deepseek-flash",
    false
  );

  assert.equal(result.ok, true);
  assert.equal(result.value.body.temperature, 0);
  assert.equal(result.value.body.top_p, 0.85);
  assert.deepEqual(result.value.warnings, []);
});

test("deepseek responses: forwards temperature and top_p when thinking is explicitly disabled", () => {
  const result = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      thinking: { type: "disabled" },
      temperature: 0.7,
      top_p: 0.9,
    },
    "deepseek-v4-pro",
    false
  );

  assert.equal(result.ok, true);
  assert.equal(result.value.body.temperature, 0.7);
  assert.equal(result.value.body.top_p, 0.9);
  assert.deepEqual(result.value.warnings, []);
});

test("deepseek responses: drops sampling controls and raises ignored warnings when thinking mode is active by default", () => {
  const result = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      temperature: 0,
      top_p: 0.5,
    },
    "deepseek-flash",
    false
  );

  assert.equal(result.ok, true);
  assert.equal("temperature" in result.value.body, false);
  assert.equal("top_p" in result.value.body, false);
  assert.deepEqual(result.value.warnings, ["temperature_ignored", "top_p_ignored"]);
});

test("deepseek responses: drops sampling controls and raises ignored warnings when reasoning effort is high", () => {
  const result = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "high" },
      temperature: 0.2,
    },
    "deepseek-flash",
    false
  );

  assert.equal(result.ok, true);
  assert.equal("temperature" in result.value.body, false);
  assert.deepEqual(result.value.warnings, ["temperature_ignored"]);
});

test("deepseek responses: rejects invalid temperature values", () => {
  const negative = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      temperature: -0.5,
    },
    "deepseek-flash",
    false
  );
  assert.equal(negative.ok, false);
  assert.equal(negative.param, "temperature");

  const tooHigh = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      temperature: 2.5,
    },
    "deepseek-flash",
    false
  );
  assert.equal(tooHigh.ok, false);
  assert.equal(tooHigh.param, "temperature");

  const nonNumeric = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      temperature: "0.5",
    },
    "deepseek-flash",
    false
  );
  assert.equal(nonNumeric.ok, false);
  assert.equal(nonNumeric.param, "temperature");
});

test("deepseek responses: rejects invalid top_p values", () => {
  const tooHigh = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      top_p: 1.2,
    },
    "deepseek-flash",
    false
  );
  assert.equal(tooHigh.ok, false);
  assert.equal(tooHigh.param, "top_p");

  const negative = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      top_p: -0.1,
    },
    "deepseek-flash",
    false
  );
  assert.equal(negative.ok, false);
  assert.equal(negative.param, "top_p");
});

test("lithos responses: forwards sampling controls when reasoning effort is none", () => {
  const result = toDeepSeekResponsesChatBody(
    {
      input: [{ type: "message", role: "user", content: "hello" }],
      reasoning: { effort: "none" },
      temperature: 0.4,
      top_p: 0.95,
    },
    "deepseek-ai/DeepSeek-V4.1-Flash",
    false,
    LITHOS_RESPONSES_PROFILE
  );

  assert.equal(result.ok, true);
  assert.equal(result.value.body.temperature, 0.4);
  assert.equal(result.value.body.top_p, 0.95);
  assert.deepEqual(result.value.warnings, []);
});
