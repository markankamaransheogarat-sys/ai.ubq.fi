import { test } from "bun:test";
import assert from "node:assert/strict";

import { meteredCodexModelRecord, openRouterCodexModels, withDeepSeekOfficialModels } from "../src/catalog/models.ts";
import { fetchOpenRouterModels, openRouterModelsSnapshot, resetOpenRouterModelsCacheForTest } from "../src/models/openrouter-models.ts";

const modelId = "~anthropic/claude-opus-latest";
const paidModel = { id: modelId, owned_by: "anthropic", supported_endpoint_types: ["openai-response"] };
const nativeEfforts = (model: Record<string, unknown>): string[] => (model.supported_reasoning_levels as { effort: string }[]).map((level) => level.effort);

const withExternalReasoning = async (levels: readonly string[], defaultEffort: string, check: () => void): Promise<void> => {
  const originalKey = process.env.OPENROUTER_API_KEY;
  process.env.OPENROUTER_API_KEY = "catalog-reasoning-order-test-key";
  resetOpenRouterModelsCacheForTest();
  try {
    await fetchOpenRouterModels({
      force: true,
      fetcher: () =>
        Promise.resolve(Response.json({ data: [{ id: modelId, reasoning: { supported_efforts: levels, default_effort: defaultEffort, mandatory: false } }] })),
    });
    check();
    assert.deepEqual(openRouterModelsSnapshot()?.models[0].reasoning?.supported_efforts, levels, "native projection does not mutate the external snapshot");
  } finally {
    resetOpenRouterModelsCacheForTest();
    if (originalKey === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = originalKey;
  }
};

test("codex native catalog: reversed external Claude efforts become ascending without adding tiers", async () => {
  await withExternalReasoning(["max", "xhigh", "high", "medium", "low"], "high", () => {
    const openRouterRow = openRouterCodexModels().find((model) => model.slug === modelId);
    assert.ok(openRouterRow);
    const paidOnlyRow = meteredCodexModelRecord(paidModel);
    for (const row of [openRouterRow, paidOnlyRow]) {
      assert.deepEqual(nativeEfforts(row), ["low", "medium", "high", "xhigh", "max"]);
      assert.equal(row.default_reasoning_level, "high");
    }
  });
});

test("codex native catalog: standard tiers sort while unknown tiers keep their strings and positions", async () => {
  const levels = ["ultra", "custom-v1", "max", "high", "CUSTOM", "xhigh", "medium", "minimal", "low", "none"];
  await withExternalReasoning(levels, "custom-v1", () => {
    for (const row of [...openRouterCodexModels(), meteredCodexModelRecord(paidModel)]) {
      assert.deepEqual(nativeEfforts(row), ["none", "custom-v1", "minimal", "low", "CUSTOM", "medium", "high", "xhigh", "max", "ultra"]);
      assert.equal(row.default_reasoning_level, "custom-v1");
    }
  });
});

test("codex native catalog: uploaded Codex effort order and default retain precedence over external metadata", async () => {
  await withExternalReasoning(["low", "medium", "high", "xhigh", "max"], "medium", () => {
    const uploadedLevels = ["max", "custom-v1", "xhigh", "high", "low"];
    const uploaded = { slug: modelId, supported_reasoning_levels: uploadedLevels, default_reasoning_level: "max" };
    const row = meteredCodexModelRecord(paidModel, uploaded);
    assert.deepEqual(nativeEfforts(row), uploadedLevels);
    assert.equal(row.default_reasoning_level, "max");
    assert.deepEqual(uploaded.supported_reasoning_levels, uploadedLevels);
  });
});

test("codex native catalog: official DeepSeek Pro tiers and default win when paid row is present first", () => {
  const originalKey = process.env.DEEPSEEK_API_KEY;
  process.env.DEEPSEEK_API_KEY = "test-deepseek-key";
  try {
    // Generic paid discovery row for deepseek-v4-pro with generic/empty reasoning tiers
    const paidRow = {
      slug: "deepseek-v4-pro",
      display_name: "DeepSeek V4 Pro (Paid)",
      owned_by: "moonshot",
      supported_endpoint_types: ["openai-response"],
      supported_reasoning_levels: [{ effort: "medium", description: "Medium" }],
      default_reasoning_level: "medium",
      context_window: 128000,
    };

    const initial = [paidRow];
    const result = withDeepSeekOfficialModels(initial);

    const proRow = result.find((model) => model.slug === "deepseek-v4-pro");
    assert.ok(proRow, "deepseek-v4-pro must be present in the catalog");

    // Must not duplicate the row
    assert.equal(result.filter((model) => model.slug === "deepseek-v4-pro").length, 1);

    // Official low/high/max tiers and default "high" must win over the generic paid row
    assert.deepEqual(
      (proRow.supported_reasoning_levels as { effort: string }[]).map((level) => level.effort),
      ["none", "low", "high", "max"],
      "official DeepSeek reasoning tiers win when paid row is present first"
    );
    assert.equal(proRow.default_reasoning_level, "high");

    // Existing context window is preserved
    assert.equal(proRow.context_window, 128000);
  } finally {
    if (originalKey === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = originalKey;
  }
});
