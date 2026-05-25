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

  it("uses the declared base model config when a GGUF repo has no config", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = input.toString();

        if (url === "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/raw/main/config.json") {
          return new Response("", { status: 404 });
        }

        if (url === "https://huggingface.co/api/models/unsloth/Qwen3.6-27B-MTP-GGUF?blobs=false") {
          return Response.json({
            tags: ["gguf", "base_model:Qwen/Qwen3.6-27B"],
            cardData: { base_model: ["Qwen/Qwen3.6-27B"] },
          });
        }

        if (url === "https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json") {
          return Response.json({
            model_type: "qwen3_5",
            text_config: {
              model_type: "qwen3_5_text",
              hidden_size: 5120,
              num_hidden_layers: 64,
              num_attention_heads: 40,
              num_key_value_heads: 8,
              head_dim: 128,
              max_position_embeddings: 262144,
              linear_key_head_dim: 128,
              linear_value_head_dim: 128,
              linear_num_key_heads: 16,
              linear_num_value_heads: 48,
              linear_conv_kernel_dim: 4,
            },
          });
        }

        return new Response("", { status: 500 });
      }),
    );

    const config = await fetchModelConfig("unsloth/Qwen3.6-27B-MTP-GGUF");

    expect(config).toMatchObject({
      model_type: "qwen3_5",
      text_config: {
        model_type: "qwen3_5_text",
        hidden_size: 5120,
        max_position_embeddings: 262144,
      },
    });
    expect(config._kvanta_warning).toEqual(expect.stringContaining("Qwen/Qwen3.6-27B"));
  });
});
