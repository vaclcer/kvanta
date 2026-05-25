import { getPrecisionBytes, type PrecisionId } from "$lib/calculation/dtypes";

export type WeightEstimate = {
  parameterCount: number;
  precision: PrecisionId;
  bytes: number;
  source: "huggingface_safetensors";
};

export function estimateWeightBytes(parameterCount: number, precision: PrecisionId): number {
  return parameterCount * getPrecisionBytes(precision);
}

export function estimateWeights(parameterCount: number, precision: PrecisionId): WeightEstimate {
  return {
    parameterCount,
    precision,
    bytes: estimateWeightBytes(parameterCount, precision),
    source: "huggingface_safetensors",
  };
}