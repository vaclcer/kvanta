export type PrecisionId =
  | "float32"
  | "float16"
  | "bfloat16"
  | "float8"
  | "int8"
  | "int4"
  | "nvfp4";

export type PrecisionOption = {
  id: PrecisionId;
  label: string;
  bytes: number;
};

export const precisionOptions: PrecisionOption[] = [
  { id: "bfloat16", label: "BF16", bytes: 2 },
  { id: "float16", label: "16", bytes: 2 },
  { id: "float8", label: "8", bytes: 1 },
  { id: "int8", label: "8", bytes: 1 },
  { id: "int4", label: "4", bytes: 0.5 },
  { id: "nvfp4", label: "4", bytes: 0.5 },
  { id: "float32", label: "FP32", bytes: 4 },
];

export const kvPrecisionOptions: PrecisionOption[] = precisionOptions.filter((option) =>
  ["float16", "float8", "int4"].includes(option.id),
);

export const weightPrecisionOptions: PrecisionOption[] = precisionOptions.filter((option) =>
  ["float16", "float8", "int4"].includes(option.id),
);

export function getPrecisionBytes(precision: PrecisionId): number {
  const match = precisionOptions.find((option) => option.id === precision);

  if (!match) {
    throw new Error(`Unsupported precision: ${precision}`);
  }

  return match.bytes;
}

export function precisionFromTorchDtype(dtype: unknown): PrecisionId | undefined {
  if (typeof dtype !== "string") {
    return undefined;
  }

  const normalized = dtype.toLowerCase();

  if (normalized.includes("bfloat16") || normalized.includes("bf16")) {
    return "bfloat16";
  }

  if (normalized.includes("float16") || normalized.includes("fp16")) {
    return "float16";
  }

  if (normalized.includes("float32") || normalized.includes("fp32")) {
    return "float32";
  }

  if (normalized.includes("float8") || normalized.includes("fp8")) {
    return "float8";
  }

  if (normalized.includes("int8")) {
    return "int8";
  }

  if (normalized.includes("int4")) {
    return "int4";
  }

  if (normalized.includes("nvfp4") || normalized.includes("fp4")) {
    return "nvfp4";
  }

  return undefined;
}