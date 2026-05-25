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

type ModelSearchCacheEntry = {
  expiresAt: number;
  value: HuggingFaceModelSearchResult[];
};

export type HuggingFaceModelInfo = {
  safetensors?: {
    total?: number;
    parameters?: Record<string, number>;
  };
};

export type HuggingFaceModelSearchResult = {
  id: string;
  pipelineTag?: string;
  trendingScore?: number;
  downloads?: number;
  likes?: number;
  gated?: boolean | "auto" | "manual";
  tags: string[];
};

type RawHuggingFaceModelSearchResult = {
  id?: unknown;
  pipeline_tag?: unknown;
  trendingScore?: unknown;
  downloads?: unknown;
  likes?: unknown;
  gated?: unknown;
  tags?: unknown;
};

const configCache = new Map<string, CacheEntry>();
const modelInfoCache = new Map<string, ModelInfoCacheEntry>();
const modelSearchCache = new Map<string, ModelSearchCacheEntry>();
const cacheTtlMs = 1000 * 60 * 10;
const searchCacheTtlMs = 1000 * 60 * 3;
const modelSearchPipelineTags = ["text-generation", "text2text-generation", "image-text-to-text"];

const knownGatedModelConfigs = new Map<string, RawModelConfig>([
  [
    "meta-llama/llama-3.1-8b",
    {
      architectures: ["LlamaForCausalLM"],
      model_type: "llama",
      hidden_size: 4096,
      intermediate_size: 14336,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      max_position_embeddings: 131072,
      rope_theta: 500000,
      torch_dtype: "bfloat16",
      _kvanta_warning:
        "Hugging Face gates the canonical config.json for this model, so kvanta used a built-in Llama 3.1 8B architecture profile.",
    },
  ],
  [
    "meta-llama/llama-3.1-8b-instruct",
    {
      architectures: ["LlamaForCausalLM"],
      model_type: "llama",
      hidden_size: 4096,
      intermediate_size: 14336,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
      max_position_embeddings: 131072,
      rope_theta: 500000,
      torch_dtype: "bfloat16",
      _kvanta_warning:
        "Hugging Face gates the canonical config.json for this model, so kvanta used a built-in Llama 3.1 8B architecture profile.",
    },
  ],
]);

function knownGatedConfigForModel(modelId: string): RawModelConfig | undefined {
  const config = knownGatedModelConfigs.get(modelId.toLowerCase());
  return config ? { ...config } : undefined;
}

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
      const fallbackConfig = knownGatedConfigForModel(trimmedModelId);

      if (fallbackConfig) {
        configCache.set(trimmedModelId, { expiresAt: Date.now() + cacheTtlMs, value: fallbackConfig });
        return fallbackConfig;
      }

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

function normalizeSearchResult(result: RawHuggingFaceModelSearchResult): HuggingFaceModelSearchResult | null {
  if (typeof result.id !== "string") {
    return null;
  }

  return {
    id: result.id,
    pipelineTag: typeof result.pipeline_tag === "string" ? result.pipeline_tag : undefined,
    trendingScore: typeof result.trendingScore === "number" ? result.trendingScore : undefined,
    downloads: typeof result.downloads === "number" ? result.downloads : undefined,
    likes: typeof result.likes === "number" ? result.likes : undefined,
    gated:
      typeof result.gated === "boolean" || result.gated === "auto" || result.gated === "manual"
        ? result.gated
        : undefined,
    tags: Array.isArray(result.tags) ? result.tags.filter((tag): tag is string => typeof tag === "string") : [],
  };
}

export async function searchHuggingFaceModels(query: string): Promise<HuggingFaceModelSearchResult[]> {
  const trimmedQuery = query.trim();

  if (trimmedQuery.length < 2) {
    return [];
  }

  const cacheKey = trimmedQuery.toLowerCase();
  const cached = modelSearchCache.get(cacheKey);

  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  const headers: HeadersInit = {};

  if (process.env.HUGGINGFACE_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HUGGINGFACE_TOKEN}`;
  }

  const responses = await Promise.all(
    modelSearchPipelineTags.map(async (pipelineTag) => {
      const searchParams = new URLSearchParams({
        search: trimmedQuery,
        pipeline_tag: pipelineTag,
        sort: "trendingScore",
        direction: "-1",
        limit: "8",
        full: "false",
      });
      const response = await fetch(`${HF_BASE_URL}/api/models?${searchParams.toString()}`, { headers });

      if (!response.ok) {
        throw new Error(`Hugging Face model search failed with status ${response.status}.`);
      }

      return (await response.json()) as RawHuggingFaceModelSearchResult[];
    }),
  );

  const resultsById = new Map<string, HuggingFaceModelSearchResult>();

  for (const result of responses.flat()) {
    const normalized = normalizeSearchResult(result);

    if (normalized) {
      resultsById.set(normalized.id, normalized);
    }
  }

  const results = [...resultsById.values()]
    .sort(
      (left, right) =>
        (right.trendingScore ?? 0) - (left.trendingScore ?? 0) ||
        (right.downloads ?? 0) - (left.downloads ?? 0) ||
        (right.likes ?? 0) - (left.likes ?? 0),
    )
    .slice(0, 12);

  modelSearchCache.set(cacheKey, { expiresAt: Date.now() + searchCacheTtlMs, value: results });

  return results;
}