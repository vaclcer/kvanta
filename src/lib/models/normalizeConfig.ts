import { precisionFromTorchDtype } from "$lib/calculation/dtypes";
import type {
  AttentionKind,
  CacheStrategy,
  NormalizedLayer,
  NormalizedModelConfig,
  RawModelConfig,
} from "$lib/models/types";

type NormalizeOptions = {
  modelId: string;
  sourceUrl: string;
};

function readNumber(config: RawModelConfig, key: string): number | undefined {
  const value = config[key];

  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  return undefined;
}

function readString(config: RawModelConfig, key: string): string | undefined {
  const value = config[key];
  return typeof value === "string" ? value : undefined;
}

function readStringArray(config: RawModelConfig, key: string): string[] {
  const value = config[key];

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string");
}

function readNumberArray(config: RawModelConfig, key: string): number[] {
  const value = config[key];

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
}

function readObject(config: RawModelConfig, key: string): RawModelConfig | undefined {
  const value = config[key];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as RawModelConfig) : undefined;
}

function decoderConfig(config: RawModelConfig): RawModelConfig {
  const textConfig = readObject(config, "text_config");
  return textConfig ? { ...config, ...textConfig } : config;
}

function inferLayerKind(layerType: string | undefined, hasSlidingWindow: boolean): AttentionKind {
  if (!layerType) {
    return hasSlidingWindow ? "sliding" : "full";
  }

  const normalized = layerType.toLowerCase();

  if (normalized.includes("sliding") || normalized.includes("local")) {
    return "sliding";
  }

  if (normalized.includes("linear")) {
    return "linear";
  }

  return "full";
}

function buildLayers(
  layerCount: number,
  slidingWindow: number | undefined,
  layerTypes: string[],
): NormalizedLayer[] {
  return Array.from({ length: layerCount }, (_, index) => {
    const attention = inferLayerKind(layerTypes[index], Boolean(slidingWindow));
    return {
      index,
      attention,
      windowSize: attention === "sliding" ? slidingWindow : undefined,
    };
  });
}

function hasMlaFields(config: RawModelConfig): boolean {
  return (
    typeof config.kv_lora_rank === "number" ||
    typeof config.q_lora_rank === "number" ||
    typeof config.qk_rope_head_dim === "number" ||
    typeof config.v_head_dim === "number"
  );
}

function buildCacheStrategy(config: RawModelConfig, modelType: string | undefined): CacheStrategy {
  if (modelType === "deepseek_v4") {
    const layerCount = readNumber(config, "num_hidden_layers");
    const headDim = readNumber(config, "head_dim");
    const slidingWindow = readNumber(config, "sliding_window");
    const indexHeadDim = readNumber(config, "index_head_dim");
    const compressRatios = readNumberArray(config, "compress_ratios");

    if (layerCount && headDim && slidingWindow && indexHeadDim && compressRatios.length >= layerCount) {
      return {
        kind: "deepseek_v4_hybrid",
        headDim,
        slidingWindow,
        indexHeadDim,
        compressRatios: compressRatios.slice(0, layerCount),
        indexerBytesPerElement: 2,
      };
    }
  }

  if (modelType === "qwen3_5_moe_text") {
    const linearKeyHeadDim = readNumber(config, "linear_key_head_dim");
    const linearValueHeadDim = readNumber(config, "linear_value_head_dim");
    const linearKeyHeads = readNumber(config, "linear_num_key_heads");
    const linearValueHeads = readNumber(config, "linear_num_value_heads");
    const linearConvKernelDim = readNumber(config, "linear_conv_kernel_dim");

    if (linearKeyHeadDim && linearValueHeadDim && linearKeyHeads && linearValueHeads && linearConvKernelDim) {
      return {
        kind: "qwen3_5_moe_hybrid",
        linearKeyHeadDim,
        linearValueHeadDim,
        linearKeyHeads,
        linearValueHeads,
        linearConvKernelDim,
        recurrentBytesPerElement: 4,
      };
    }
  }

  if (modelType === "glm_moe_dsa") {
    const keyHeadDim = readNumber(config, "qk_head_dim");
    const valueHeadDim = readNumber(config, "v_head_dim");
    const indexHeadDim = readNumber(config, "index_head_dim");

    if (keyHeadDim && valueHeadDim && indexHeadDim) {
      return {
        kind: "glm_moe_dsa_expanded",
        keyHeadDim,
        valueHeadDim,
        indexHeadDim,
      };
    }
  }

  return { kind: "standard" };
}

export function normalizeConfig(
  config: RawModelConfig,
  options: NormalizeOptions,
): NormalizedModelConfig {
  const originalConfig = config;
  config = decoderConfig(config);
  const warnings: string[] = [];
  const unsupportedReasons: string[] = [];
  const hiddenSize = readNumber(config, "hidden_size");
  const numHiddenLayers = readNumber(config, "num_hidden_layers");
  const numAttentionHeads = readNumber(config, "num_attention_heads");
  const explicitKvHeads = readNumber(config, "num_key_value_heads");
  const configuredHeadDim = readNumber(config, "head_dim");
  const slidingWindow = readNumber(config, "sliding_window");
  const layerTypes = readStringArray(config, "layer_types");
  const architectures = readStringArray(config, "architectures");
  const modelType = readString(config, "model_type");
  const cacheStrategy = buildCacheStrategy(config, modelType);

  if (config.is_encoder_decoder === true) {
    unsupportedReasons.push("Encoder-decoder KV cache formulas are architecture-specific and not enabled yet.");
  }

  if (hasMlaFields(config) && cacheStrategy.kind === "standard") {
    unsupportedReasons.push("MLA/compressed-cache architecture detected. Exact adapter support is required before calculating.");
  }

  if (cacheStrategy.kind === "glm_moe_dsa_expanded") {
    warnings.push(
      "GLM MoE DSA uses the Hugging Face Transformers cache layout: expanded attention K/V plus a per-layer DSA indexer key cache.",
    );
  }

  if (cacheStrategy.kind === "deepseek_v4_hybrid") {
    warnings.push(
      "DeepSeek V4 uses the official hybrid sparse cache layout: sliding latent KV, ratio-compressed latent KV, and FP16 indexer cache for ratio-4 layers.",
    );
  }

  if (cacheStrategy.kind === "qwen3_5_moe_hybrid") {
    warnings.push(
      "Qwen3.5 MoE uses a hybrid cache: normal K/V for full-attention layers plus fixed linear-attention convolution and recurrent states.",
    );
  }

  if (!hiddenSize) {
    unsupportedReasons.push("Missing numeric hidden_size in Hugging Face config.");
  }

  if (!numHiddenLayers) {
    unsupportedReasons.push("Missing numeric num_hidden_layers in Hugging Face config.");
  }

  if (!numAttentionHeads) {
    unsupportedReasons.push("Missing numeric num_attention_heads in Hugging Face config.");
  }

  const numKeyValueHeads = explicitKvHeads ?? numAttentionHeads ?? 0;

  if (!explicitKvHeads && numAttentionHeads) {
    warnings.push("num_key_value_heads is missing; using num_attention_heads as the exact fallback for non-GQA models.");
  }

  let headDim = configuredHeadDim;

  if (!headDim && hiddenSize && numAttentionHeads) {
    if (hiddenSize % numAttentionHeads === 0) {
      headDim = hiddenSize / numAttentionHeads;
      warnings.push("head_dim is missing; computed hidden_size / num_attention_heads.");
    } else {
      unsupportedReasons.push("Cannot derive head_dim because hidden_size is not divisible by num_attention_heads.");
    }
  }

  if (slidingWindow && layerTypes.length === 0) {
    warnings.push("sliding_window is set and no layer_types array was found; applying the window to every layer.");
  }

  const layers = buildLayers(numHiddenLayers ?? 0, slidingWindow, layerTypes);

  return {
    modelId: options.modelId,
    sourceUrl: options.sourceUrl,
    modelType,
    architectures,
    hiddenSize: hiddenSize ?? 0,
    numHiddenLayers: numHiddenLayers ?? 0,
    numAttentionHeads: numAttentionHeads ?? 0,
    numKeyValueHeads,
    headDim: headDim ?? 0,
    cacheStrategy,
    maxPositionEmbeddings: readNumber(config, "max_position_embeddings"),
    defaultPrecision:
      cacheStrategy.kind === "deepseek_v4_hybrid"
        ? "float8"
        : precisionFromTorchDtype(config.torch_dtype ?? config.dtype),
    layers,
    warnings,
    unsupportedReasons,
    rawConfig: originalConfig,
  };
}