import { getPrecisionBytes } from "$lib/calculation/dtypes";
import type { CalculateResult, LayerBreakdown, NormalizedModelConfig } from "$lib/models/types";
import type { PrecisionId } from "$lib/calculation/dtypes";

type CalculateKvCacheOptions = {
  model: NormalizedModelConfig;
  sequenceLength: number;
  batchSize: number;
  precision: PrecisionId;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

export function calculateKvCache({
  model,
  sequenceLength,
  batchSize,
  precision,
}: CalculateKvCacheOptions): CalculateResult {
  assertPositiveInteger("sequenceLength", sequenceLength);
  assertPositiveInteger("batchSize", batchSize);

  if (model.unsupportedReasons.length > 0) {
    throw new Error(model.unsupportedReasons.join(" "));
  }

  const precisionBytes = getPrecisionBytes(precision);
  const layerBreakdown: LayerBreakdown[] = model.layers.map((layer) => {
    const tokens =
      layer.attention === "sliding" && layer.windowSize
        ? Math.min(sequenceLength, layer.windowSize)
        : sequenceLength;

    if (model.cacheStrategy.kind === "glm_moe_dsa_expanded") {
      const keyBytes =
        batchSize * tokens * model.numAttentionHeads * model.cacheStrategy.keyHeadDim * precisionBytes;
      const valueBytes =
        batchSize * tokens * model.numAttentionHeads * model.cacheStrategy.valueHeadDim * precisionBytes;
      const indexerBytes = batchSize * tokens * model.cacheStrategy.indexHeadDim * precisionBytes;
      const components: Record<string, number> = {
        keyBytes,
        valueBytes,
        indexerBytes,
      };

      return {
        index: layer.index,
        attention: layer.attention,
        tokens,
        bytes: keyBytes + valueBytes + indexerBytes,
        components,
      };
    }

    if (model.cacheStrategy.kind === "deepseek_v4_hybrid") {
      const compressRatio = model.cacheStrategy.compressRatios[layer.index] ?? 0;
      const slidingTokens = model.cacheStrategy.slidingWindow;
      const compressedTokens = compressRatio > 0 ? Math.floor(sequenceLength / compressRatio) : 0;
      const slidingKvBytes =
        batchSize * slidingTokens * model.cacheStrategy.headDim * precisionBytes;
      const compressedKvBytes =
        batchSize * compressedTokens * model.cacheStrategy.headDim * precisionBytes;
      const indexerBytes =
        compressRatio === 4
          ? batchSize * compressedTokens * model.cacheStrategy.indexHeadDim * model.cacheStrategy.indexerBytesPerElement
          : 0;
      const components: Record<string, number> = {
        slidingKvBytes,
        compressedKvBytes,
        indexerBytes,
      };

      return {
        index: layer.index,
        attention: layer.attention,
        tokens: slidingTokens + compressedTokens,
        bytes: slidingKvBytes + compressedKvBytes + indexerBytes,
        components,
      };
    }

    if (model.cacheStrategy.kind === "qwen3_5_moe_hybrid" && layer.attention === "linear") {
      const convElements =
        (model.cacheStrategy.linearKeyHeadDim * model.cacheStrategy.linearKeyHeads * 2 +
          model.cacheStrategy.linearValueHeadDim * model.cacheStrategy.linearValueHeads) *
        model.cacheStrategy.linearConvKernelDim;
      const recurrentElements =
        model.cacheStrategy.linearValueHeads *
        model.cacheStrategy.linearKeyHeadDim *
        model.cacheStrategy.linearValueHeadDim;
      const convBytes = batchSize * convElements * precisionBytes;
      const recurrentBytes = batchSize * recurrentElements * model.cacheStrategy.recurrentBytesPerElement;
      const components: Record<string, number> = {
        convBytes,
        recurrentBytes,
      };

      return {
        index: layer.index,
        attention: layer.attention,
        tokens: 1,
        bytes: convBytes + recurrentBytes,
        components,
      };
    }

    const bytes = batchSize * tokens * model.numKeyValueHeads * model.headDim * 2 * precisionBytes;

    return {
      index: layer.index,
      attention: layer.attention,
      tokens,
      bytes,
    };
  });
  const totalBytes = layerBreakdown.reduce((sum, layer) => sum + layer.bytes, 0);
  const perTokenBytes = totalBytes / sequenceLength / batchSize;

  return {
    totalBytes,
    perTokenBytes,
    layerBreakdown,
    model: {
      modelId: model.modelId,
      sourceUrl: model.sourceUrl,
      modelType: model.modelType,
      architectures: model.architectures,
      hiddenSize: model.hiddenSize,
      numHiddenLayers: model.numHiddenLayers,
      numAttentionHeads: model.numAttentionHeads,
      numKeyValueHeads: model.numKeyValueHeads,
      headDim: model.headDim,
      cacheStrategy: model.cacheStrategy,
      maxPositionEmbeddings: model.maxPositionEmbeddings,
      defaultPrecision: model.defaultPrecision,
      layers: model.layers,
      warnings: model.warnings,
      unsupportedReasons: model.unsupportedReasons,
    },
    assumptions:
      model.cacheStrategy.kind === "deepseek_v4_hybrid"
        ? [
            `DeepSeek V4 hybrid cache uses latent cache elements from head_dim, not expanded key/value tensors.`,
            `KV precision ${precision} uses ${precisionBytes} bytes per latent KV element.`,
            `Indexer cache uses FP16 storage at ${model.cacheStrategy.indexerBytesPerElement} bytes per element.`,
          ]
        : model.cacheStrategy.kind === "qwen3_5_moe_hybrid"
          ? [
              `Full-attention layers store normal key/value tensors.`,
              `Linear-attention layers store a fixed convolution state at ${precision} plus an FP32 recurrent state.`,
            ]
        : [
            `KV cache stores both key and value tensors, so the formula includes a factor of 2.`,
            `Precision ${precision} uses ${precisionBytes} bytes per KV element.`,
          ],
    warnings: model.warnings,
  };
}