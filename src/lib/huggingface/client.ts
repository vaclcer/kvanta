import type { RawModelConfig } from "$lib/models/types";

const HF_BASE_URL = "https://huggingface.co";

type CacheEntry = {
  expiresAt: number;
  value: RawModelConfig;
};

type ModelInfoCacheEntry = {
  expiresAt: number;
  value: HuggingFaceModelInfo;
};

export type HuggingFaceModelInfo = {
  safetensors?: {
    total?: number;
    parameters?: Record<string, number>;
  };
};

const configCache = new Map<string, CacheEntry>();
const modelInfoCache = new Map<string, ModelInfoCacheEntry>();
const cacheTtlMs = 1000 * 60 * 10;

function encodeModelId(modelId: string): string {
  return modelId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

export function configSourceUrl(modelId: string): string {
  return `${HF_BASE_URL}/${encodeModelId(modelId)}/raw/main/config.json`;
}

export async function fetchModelConfig(modelId: string): Promise<RawModelConfig> {
  const trimmedModelId = modelId.trim();

  if (!trimmedModelId || !trimmedModelId.includes("/")) {
    throw new Error("Model ID must look like owner/model-name.");
  }

  const cached = configCache.get(trimmedModelId);

  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  const headers: HeadersInit = {};

  if (process.env.HUGGINGFACE_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HUGGINGFACE_TOKEN}`;
  }

  const response = await fetch(configSourceUrl(trimmedModelId), {
    headers,
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error("Hugging Face rejected the request. This model may be gated or private.");
    }

    if (response.status === 404) {
      throw new Error("No config.json was found for that model ID.");
    }

    throw new Error(`Hugging Face config fetch failed with status ${response.status}.`);
  }

  const config = (await response.json()) as RawModelConfig;
  configCache.set(trimmedModelId, { expiresAt: Date.now() + cacheTtlMs, value: config });

  return config;
}

export async function fetchModelInfo(modelId: string): Promise<HuggingFaceModelInfo> {
  const trimmedModelId = modelId.trim();

  if (!trimmedModelId || !trimmedModelId.includes("/")) {
    throw new Error("Model ID must look like owner/model-name.");
  }

  const cached = modelInfoCache.get(trimmedModelId);

  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  const headers: HeadersInit = {};

  if (process.env.HUGGINGFACE_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HUGGINGFACE_TOKEN}`;
  }

  const response = await fetch(`https://huggingface.co/api/models/${trimmedModelId}?blobs=false`, {
    headers,
  });

  if (!response.ok) {
    throw new Error(`Hugging Face model metadata fetch failed with status ${response.status}.`);
  }

  const info = (await response.json()) as HuggingFaceModelInfo;
  modelInfoCache.set(trimmedModelId, { expiresAt: Date.now() + cacheTtlMs, value: info });

  return info;
}