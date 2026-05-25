import { describe, expect, it } from "vitest";
import { calculateKvCache } from "$lib/calculation/kvCache";
import { normalizeConfig } from "$lib/models/normalizeConfig";

function normalize(config: Record<string, unknown>) {
  return normalizeConfig(config, {
    modelId: "test/model",
    sourceUrl: "https://huggingface.co/test/model/raw/main/config.json",
  });
}

describe("calculateKvCache", () => {
  it("calculates full-attention KV cache from layer count and KV heads", () => {
    const model = normalize({
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 4096,
      batchSize: 2,
      precision: "bfloat16",
    });

    expect(result.totalBytes).toBe(2 * 32 * 4096 * 8 * 128 * 2 * 2);
  });

  it("caps sliding-window layers at the configured window size", () => {
    const model = normalize({
      hidden_size: 4096,
      num_hidden_layers: 2,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      sliding_window: 1024,
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 4096,
      batchSize: 1,
      precision: "float16",
    });

    expect(result.layerBreakdown.every((layer) => layer.tokens === 1024)).toBe(true);
    expect(result.totalBytes).toBe(2 * 1024 * 8 * 128 * 2 * 2);
  });

  it("uses layer_types to mix full and sliding attention", () => {
    const model = normalize({
      hidden_size: 1024,
      num_hidden_layers: 2,
      num_attention_heads: 8,
      num_key_value_heads: 2,
      head_dim: 128,
      sliding_window: 512,
      layer_types: ["full_attention", "sliding_attention"],
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 2048,
      batchSize: 1,
      precision: "float8",
    });

    expect(result.layerBreakdown.map((layer) => layer.tokens)).toEqual([2048, 512]);
    expect(result.totalBytes).toBe((2048 + 512) * 2 * 128 * 2 * 1);
  });

  it("calculates GLM MoE DSA expanded K/V and indexer cache", () => {
    const model = normalize({
      dtype: "bfloat16",
      model_type: "glm_moe_dsa",
      hidden_size: 6144,
      num_hidden_layers: 78,
      num_attention_heads: 64,
      num_key_value_heads: 64,
      head_dim: 64,
      qk_head_dim: 256,
      qk_nope_head_dim: 192,
      qk_rope_head_dim: 64,
      kv_lora_rank: 512,
      v_head_dim: 256,
      index_head_dim: 128,
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 4096,
      batchSize: 1,
      precision: "bfloat16",
    });
    const keyBytes = 4096 * 64 * 256 * 2;
    const valueBytes = 4096 * 64 * 256 * 2;
    const indexerBytes = 4096 * 128 * 2;

    expect(result.model.cacheStrategy.kind).toBe("glm_moe_dsa_expanded");
    expect(result.layerBreakdown[0].components).toEqual({ keyBytes, valueBytes, indexerBytes });
    expect(result.totalBytes).toBe(78 * (keyBytes + valueBytes + indexerBytes));
  });

  it("calculates DeepSeek V4 hybrid sparse cache from compress ratios", () => {
    const compressRatios = Array.from({ length: 61 }, (_, index) => {
      if (index < 2) {
        return 128;
      }

      return index % 2 === 0 ? 4 : 128;
    });
    const model = normalize({
      model_type: "deepseek_v4",
      hidden_size: 7168,
      num_hidden_layers: 61,
      num_attention_heads: 128,
      num_key_value_heads: 1,
      head_dim: 512,
      q_lora_rank: 1536,
      qk_rope_head_dim: 64,
      sliding_window: 128,
      index_head_dim: 128,
      compress_ratios: [...compressRatios, 0],
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 1024,
      batchSize: 1,
      precision: "float8",
    });

    expect(result.model.cacheStrategy.kind).toBe("deepseek_v4_hybrid");
    expect(result.totalBytes).toBe(10_022_912);
    expect(result.layerBreakdown.reduce((sum, layer) => sum + (layer.components?.indexerBytes ?? 0), 0)).toBe(
      1_966_080,
    );
  });

  it("calculates nested Qwen3.5 MoE hybrid full and linear attention cache", () => {
    const layerTypes = Array.from({ length: 48 }, (_, index) =>
      (index + 1) % 4 === 0 ? "full_attention" : "linear_attention",
    );
    const model = normalize({
      model_type: "qwen3_5_moe",
      text_config: {
        dtype: "bfloat16",
        model_type: "qwen3_5_moe_text",
        hidden_size: 3072,
        num_hidden_layers: 48,
        num_attention_heads: 32,
        num_key_value_heads: 2,
        head_dim: 256,
        layer_types: layerTypes,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_num_key_heads: 16,
        linear_num_value_heads: 64,
        linear_conv_kernel_dim: 4,
      },
    });

    const result = calculateKvCache({
      model,
      sequenceLength: 1024,
      batchSize: 1,
      precision: "bfloat16",
    });

    const linearLayerBytes = 98_304 + 4_194_304;
    const fullLayerBytes = 1024 * 2 * 256 * 2 * 2;

    expect(result.model.cacheStrategy.kind).toBe("qwen3_5_moe_hybrid");
    expect(result.layerBreakdown.filter((layer) => layer.attention === "linear")).toHaveLength(36);
    expect(result.layerBreakdown[0].components).toEqual({
      convBytes: 98_304,
      recurrentBytes: 4_194_304,
    });
    expect(result.totalBytes).toBe(36 * linearLayerBytes + 12 * fullLayerBytes);
  });

  it("rejects MLA configs until an exact adapter exists", () => {
    const model = normalize({
      hidden_size: 7168,
      num_hidden_layers: 61,
      num_attention_heads: 128,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
    });

    expect(() =>
      calculateKvCache({
        model,
        sequenceLength: 4096,
        batchSize: 1,
        precision: "float8",
      }),
    ).toThrow("MLA/compressed-cache architecture detected");
  });
});