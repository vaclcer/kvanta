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

  if (normalized.includes("mamba") || normalized.includes("recurrent") || normalized.includes("ssm")) {
    return "recurrent";
  }

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
  forcedKind?: AttentionKind,
): NormalizedLayer[] {
  return Array.from({ length: layerCount }, (_, index) => {
    const attention = forcedKind ?? inferLayerKind(layerTypes[index], Boolean(slidingWindow));
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
    const kvLoraRank = readNumber(config, "kv_lora_rank");
    const qkRopeHeadDim = readNumber(config, "qk_rope_head_dim");
    const indexHeadDim = readNumber(config, "index_head_dim");

    if (kvLoraRank && qkRopeHeadDim && indexHeadDim) {
      return {
        kind: "glm_moe_dsa_compressed",
        kvLoraRank,
        qkRopeHeadDim,
        indexHeadDim,
      };
    }
  }
if (modelType === "mamba2") {
    const hiddenSize = readNumber(config, "hidden_size") ?? 0;
    const expand = readNumber(config, "expand") ?? 2;
    const intermediateSize = readNumber(config, "intermediate_size") ?? hiddenSize * expand;
    const stateSize = readNumber(config, "state_size") ?? 128;
    const convKernel = readNumber(config, "conv_kernel") ?? 4;
    const numHeads = readNumber(config, "num_heads");
    const headDim = readNumber(config, "head_dim");
    const nGroups = readNumber(config, "n_groups") ?? 1;

    if (intermediateSize && stateSize && convKernel && numHeads && headDim) {
      return {
        kind: "mamba2_ssm",
        intermediateSize,
        stateSize,
        convKernel,
        numHeads,
        headDim,
        nGroups,
        recurrentBytesPerElement: 4,
      };
    }
  }

  if (modelType === "mamba" || modelType === "falcon_mamba") {
    const hiddenSize = readNumber(config, "hidden_size") ?? 0;
    const expand = readNumber(config, "expand") ?? 2;
    const intermediateSize = readNumber(config, "intermediate_size") ?? hiddenSize * expand;
    const stateSize = readNumber(config, "state_size") ?? 16;
    const convKernel = readNumber(config, "conv_kernel") ?? 4;

    if (intermediateSize && stateSize && convKernel) {
      return {
        kind: "mamba_ssm",
        intermediateSize,
        stateSize,
        convKernel,
        recurrentBytesPerElement: 4,
      };
    }
  }

  const kvLoraRank = readNumber(config, "kv_lora_rank");
  const qkRopeHeadDim = readNumber(config, "qk_rope_head_dim");

  if (kvLoraRank && qkRopeHeadDim) {
    return {
      kind: "mla_compressed",
      kvLoraRank,
      qkRopeHeadDim,
    };
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
  const kvantaWarning = readString(config, "_kvanta_warning");

  if (kvantaWarning) {
    warnings.push(kvantaWarning);
  }

  if (config.is_encoder_decoder === true) {
    unsupportedReasons.push("Encoder-decoder KV cache formulas are architecture-specific and not enabled yet.");
  }

  if (hasMlaFields(config) && cacheStrategy.kind === "standard") {
    unsupportedReasons.push("MLA/compressed-cache architecture detected but missing required fields (kv_lora_rank + qk_rope_head_dim).");
  }

  if (cacheStrategy.kind === "glm_moe_dsa_compressed") {
    warnings.push(
      "GLM MoE DSA uses an optimized compressed MLA cache plus a per-layer DSA indexer key cache.",
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

  if (cacheStrategy.kind === "mla_compressed") {
    warnings.push(
      "MLA compressed cache stores the latent KV vector (kv_lora_rank) plus the RoPE key (qk_rope_head_dim) per token per layer. Matches vLLM / SGLang / TensorRT-LLM compressed storage.",
    );
  }

  if (cacheStrategy.kind === "mamba_ssm" || cacheStrategy.kind === "mamba2_ssm") {
    warnings.push(
      "Mamba / SSM cache is a fixed-size per-layer state that does not grow with sequence length.",
    );
  }

  if (!hiddenSize) {
    unsupportedReasons.push("Missing numeric hidden_size in Hugging Face config.");
  }

  if (!numHiddenLayers) {
    unsupportedReasons.push("Missing numeric num_hidden_layers in Hugging Face config.");
  }

  const isMambaStrategy =
    cacheStrategy.kind === "mamba_ssm" || cacheStrategy.kind === "mamba2_ssm";
  const isMlaStrategy = cacheStrategy.kind === "mla_compressed";
  const headDimOptional = isMambaStrategy || isMlaStrategy;

  if (!numAttentionHeads && !isMambaStrategy) {
    unsupportedReasons.push("Missing numeric num_attention_heads in Hugging Face config.");
  }

  const numKeyValueHeads = explicitKvHeads ?? numAttentionHeads ?? 0;

  if (!explicitKvHeads && numAttentionHeads && !isMambaStrategy && !isMlaStrategy) {
    warnings.push("num_key_value_heads is missing; using num_attention_heads as the exact fallback for non-GQA models.");
  }

  let headDim = configuredHeadDim;

  if (!headDim && hiddenSize && numAttentionHeads && !headDimOptional) {
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

  const forcedLayerKind: AttentionKind | undefined = isMambaStrategy ? "recurrent" : undefined;
  const layers = buildLayers(numHiddenLayers ?? 0, slidingWindow, layerTypes, forcedLayerKind);

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