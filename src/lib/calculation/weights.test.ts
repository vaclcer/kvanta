import { describe, expect, it } from "vitest";
import { estimateWeightBytes, estimateWeights } from "$lib/calculation/weights";

describe("weight estimates", () => {
  it("estimates parameter storage from quantization bytes", () => {
    expect(estimateWeightBytes(1_000_000_000, "bfloat16")).toBe(2_000_000_000);
    expect(estimateWeightBytes(1_000_000_000, "float8")).toBe(1_000_000_000);
    expect(estimateWeightBytes(1_000_000_000, "int4")).toBe(500_000_000);
    expect(estimateWeightBytes(1_000_000_000, "nvfp4")).toBe(500_000_000);
  });

  it("includes the source and precision in the estimate", () => {
    expect(estimateWeights(123, "int8")).toEqual({
      parameterCount: 123,
      precision: "int8",
      bytes: 123,
      source: "huggingface_safetensors",
    });
  });
});