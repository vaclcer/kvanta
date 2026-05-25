import type { PrecisionId } from "$lib/calculation/dtypes";
import type { WeightEstimate } from "$lib/calculation/weights";

export type RawModelConfig = Record<string, unknown>;

export type AttentionKind = "full" | "sliding" | "linear";

export type NormalizedLayer = {
  index: number;
  attention: AttentionKind;
  windowSize?: number;
};

export type CacheStrategy =
  | {
      kind: "standard";
    }
  | {
      kind: "glm_moe_dsa_expanded";
      keyHeadDim: number;
      valueHeadDim: number;
      indexHeadDim: number;
    }
  | {
      kind: "deepseek_v4_hybrid";
      headDim: number;
      slidingWindow: number;
      indexHeadDim: number;
      compressRatios: number[];
      indexerBytesPerElement: number;
    }
  | {
      kind: "qwen3_5_moe_hybrid";
      linearKeyHeadDim: number;
      linearValueHeadDim: number;
      linearKeyHeads: number;
      linearValueHeads: number;
      linearConvKernelDim: number;
      recurrentBytesPerElement: number;
    };

export type NormalizedModelConfig = {
  modelId: string;
  sourceUrl: string;
  modelType?: string;
  architectures: string[];
  hiddenSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  cacheStrategy: CacheStrategy;
  maxPositionEmbeddings?: number;
  defaultPrecision?: PrecisionId;
  layers: NormalizedLayer[];
  warnings: string[];
  unsupportedReasons: string[];
  rawConfig: RawModelConfig;
};

export type CalculateRequest = {
  modelId?: string;
  config?: RawModelConfig;
  sequenceLength: number;
  batchSize: number;
  precision: PrecisionId;
  weightPrecision?: PrecisionId;
};

export type LayerBreakdown = {
  index: number;
  attention: AttentionKind;
  tokens: number;
  bytes: number;
  components?: Record<string, number>;
};

export type CalculateResult = {
  totalBytes: number;
  perTokenBytes: number;
  layerBreakdown: LayerBreakdown[];
  model: Omit<NormalizedModelConfig, "rawConfig">;
  weights?: WeightEstimate;
  assumptions: string[];
  warnings: string[];
};