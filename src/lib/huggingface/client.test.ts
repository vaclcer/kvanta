import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchModelConfig } from "$lib/huggingface/client";

function mockFetch(status: number) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response("", { status })),
  );
}

describe("Hugging Face client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses the built-in Llama 3.1 8B config when the gated config is rejected", async () => {
    mockFetch(401);

    const config = await fetchModelConfig("meta-llama/Llama-3.1-8B");

    expect(config).toMatchObject({
      model_type: "llama",
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      max_position_embeddings: 131072,
    });
    expect(config._kvanta_warning).toEqual(expect.stringContaining("built-in Llama 3.1 8B"));
  });

  it("still rejects unknown gated models", async () => {
    mockFetch(401);

    await expect(fetchModelConfig("some-owner/some-gated-model")).rejects.toThrow("gated or private");
  });
});
