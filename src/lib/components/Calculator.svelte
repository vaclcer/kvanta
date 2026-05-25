<script lang="ts">
  import { onMount } from "svelte";
  import { calculateKvCache } from "$lib/calculation/kvCache";
  import { formatBytes, formatInteger } from "$lib/calculation/format";
  import {
    getPrecisionBytes,
    kvPrecisionOptions,
    weightPrecisionOptions,
    type PrecisionId,
  } from "$lib/calculation/dtypes";
  import type { CalculateResult, NormalizedModelConfig } from "$lib/models/types";

  type CalculateResponse = {
    result?: CalculateResult;
    error?: string;
  };

  type ModelSearchResult = {
    id: string;
    pipelineTag?: string;
    trendingScore?: number;
    downloads?: number;
    likes?: number;
    gated?: boolean | "auto" | "manual";
    tags: string[];
  };

  type ModelSearchResponse = {
    results?: ModelSearchResult[];
    error?: string;
  };

  type ProcessedModel = {
    id: string;
    selected: boolean;
    result?: CalculateResult;
    error?: string;
    loading?: boolean;
  };

  type GraphPoint = {
    context: number;
    kvBytes: number;
    weightBytes: number;
    bytes: number;
  };

  type GraphSeries = {
    id: string;
    color: string;
    points: GraphPoint[];
  };

  type GraphScale = {
    minBytes: number;
    maxBytes: number;
    yTicks: number[];
    xTicks: number[];
  };

  type HoveredGraph = {
    context: number;
    x: number;
    items: Array<{
      id: string;
      color: string;
      kvBytes: number;
      weightBytes: number;
      bytes: number;
      y: number;
    }>;
  };

  type GraphMode = "kv" | "total";

  const processedModelsKey = "kvanta:processed-models";
  const palette = ["#111111", "#2f6df6", "#d45f00", "#15803d", "#7c3aed", "#be123c"];

  let modelInput = $state("");
  let processedModels = $state<ProcessedModel[]>([]);
  let sequenceLength = $state(32768);
  let batchSize = $state(1);
  let precision = $state<PrecisionId>("float16");
  let weightPrecision = $state<PrecisionId>("float16");
  let graphMode = $state<GraphMode>("total");
  let activeTab = $state<"graph" | "architecture">("graph");
  let pickerError = $state<string | null>(null);
  let modelSearchResults = $state<ModelSearchResult[]>([]);
  let modelSearchLoading = $state(false);
  let modelSearchError = $state<string | null>(null);
  let hoveredGraph = $state<HoveredGraph | null>(null);
  let modelSearchTimer: ReturnType<typeof setTimeout> | undefined;
  let modelSearchRequestId = 0;

  const selectedModels = $derived(processedModels.filter((model) => model.selected && model.result));
  const graphSeries = $derived(buildGraphSeries(selectedModels));
  const graphMaxContext = $derived(Math.max(1, sequenceLength));
  const graphScale = $derived(buildGraphScale(graphSeries));
  const tooltipWidth = $derived(Math.min(390, Math.max(250, 190 + hoveredGraphItemsMaxLength(hoveredGraph) * 8)));

  function graphBytes(kvBytes: number, weightBytes: number): number {
    return graphMode === "kv" ? kvBytes : kvBytes + weightBytes;
  }

  function hoveredGraphItemsMaxLength(graph: HoveredGraph | null): number {
    return graph ? Math.max(...graph.items.map((item) => modelName(item.id).length), 0) : 0;
  }

  function tooltipX(x: number): number {
    return Math.min(760 - tooltipWidth, Math.max(54, x + 14));
  }

  function compactName(modelId: string): string {
    const name = modelName(modelId);
    return name.length > 28 ? `${name.slice(0, 25)}...` : name;
  }

  function parseHuggingFaceInput(value: string): string {
    const trimmed = value.trim().replace(/\/+$/, "");

    if (!trimmed) {
      return "";
    }

    try {
      const url = new URL(trimmed);

      if (!url.hostname.endsWith("huggingface.co")) {
        return trimmed;
      }

      const [owner, repo] = url.pathname.split("/").filter(Boolean);
      return owner && repo ? `${owner}/${repo}` : "";
    } catch {
      return trimmed.replace(/^huggingface\.co\//, "");
    }
  }

  function isValidModelId(value: string): boolean {
    const [owner, repo, ...rest] = value.split("/");
    return Boolean(owner && repo && rest.length === 0);
  }

  function shouldSearchModels(value: string): boolean {
    const trimmed = value.trim();
    return trimmed.length >= 2 && !trimmed.includes("huggingface.co") && !trimmed.startsWith("http");
  }

  function searchResultMeta(result: ModelSearchResult): string {
    const parts = [
      result.pipelineTag,
      result.trendingScore !== undefined ? `${formatInteger(result.trendingScore)} trending` : undefined,
      result.downloads !== undefined ? `${formatInteger(result.downloads)} downloads` : undefined,
    ];

    if (result.gated) {
      parts.push("gated");
    }

    return parts.filter(Boolean).join(" / ");
  }

  function handleModelInput(value: string) {
    modelInput = value;
    pickerError = null;
    modelSearchError = null;

    if (modelSearchTimer) {
      clearTimeout(modelSearchTimer);
    }

    if (!shouldSearchModels(value)) {
      modelSearchResults = [];
      modelSearchLoading = false;
      modelSearchRequestId += 1;
      return;
    }

    modelSearchLoading = true;
    const requestId = modelSearchRequestId + 1;
    modelSearchRequestId = requestId;
    modelSearchTimer = setTimeout(() => {
      void searchModels(value, requestId);
    }, 250);
  }

  async function searchModels(query: string, requestId: number) {
    try {
      const response = await fetch(`/api/model-search?q=${encodeURIComponent(query.trim())}`);
      const data = (await response.json()) as ModelSearchResponse;

      if (requestId !== modelSearchRequestId) {
        return;
      }

      if (!response.ok) {
        throw new Error(data.error ?? "Model search failed.");
      }

      modelSearchResults = data.results ?? [];
      modelSearchError = null;
    } catch (caught) {
      if (requestId !== modelSearchRequestId) {
        return;
      }

      modelSearchResults = [];
      modelSearchError = caught instanceof Error ? caught.message : "Model search failed.";
    } finally {
      if (requestId === modelSearchRequestId) {
        modelSearchLoading = false;
      }
    }
  }

  function storagePayload() {
    return processedModels.map((model) => ({ id: model.id, selected: model.selected }));
  }

  function saveProcessedModels() {
    localStorage.setItem(processedModelsKey, JSON.stringify(storagePayload()));
  }

  async function loadProcessedModels() {
    const stored = localStorage.getItem(processedModelsKey);

    if (!stored) {
      return;
    }

    try {
      const parsed = JSON.parse(stored) as Array<{ id?: unknown; selected?: unknown }>;

      if (!Array.isArray(parsed)) {
        return;
      }

      const restored = parsed
        .filter((item): item is { id: string; selected?: boolean } => typeof item.id === "string")
        .filter((item) => isValidModelId(item.id))
        .slice(0, 12)
        .map((item) => ({ id: item.id, selected: item.selected !== false, loading: true }));

      processedModels = restored;
      await Promise.all(restored.map((model) => calculateModel(model.id)));
    } catch {
      localStorage.removeItem(processedModelsKey);
    }
  }

  async function addModel() {
    const modelId = parseHuggingFaceInput(modelInput);

    await addModelById(modelId);
  }

  async function addModelById(modelId: string) {
    pickerError = null;
    modelSearchResults = [];
    modelSearchError = null;

    if (!isValidModelId(modelId)) {
      pickerError = "Paste a Hugging Face URL or enter an ID like owner/model-name.";
      return;
    }

    const existing = processedModels.find((model) => model.id === modelId);

    if (existing) {
      processedModels = processedModels.map((model) =>
        model.id === modelId ? { ...model, selected: true } : model,
      );
      saveProcessedModels();
      modelInput = "";
      return;
    }

    processedModels = [{ id: modelId, selected: true, loading: true }, ...processedModels].slice(0, 12);
    saveProcessedModels();
    modelInput = "";
    await calculateModel(modelId);
  }

  async function addSearchResult(modelId: string) {
    modelInput = modelId;
    await addModelById(modelId);
  }

  async function calculateModel(modelId: string) {
    processedModels = processedModels.map((model) =>
      model.id === modelId ? { ...model, loading: true, error: undefined } : model,
    );

    try {
      const response = await fetch("/api/calculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelId, sequenceLength, batchSize, precision, weightPrecision }),
      });
      const data = (await response.json()) as CalculateResponse;

      if (!response.ok || !data.result) {
        throw new Error(data.error ?? "Calculation failed.");
      }

      processedModels = processedModels.map((model) =>
        model.id === modelId ? { ...model, result: data.result, loading: false, error: undefined } : model,
      );
      saveProcessedModels();
    } catch (caught) {
      processedModels = processedModels.map((model) =>
        model.id === modelId
          ? {
              ...model,
              loading: false,
              error: caught instanceof Error ? caught.message : "Calculation failed.",
            }
          : model,
      );
    }
  }

  function removeModel(modelId: string) {
    processedModels = processedModels.filter((model) => model.id !== modelId);
    saveProcessedModels();
  }

  function toggleModel(modelId: string) {
    processedModels = processedModels.map((model) =>
      model.id === modelId ? { ...model, selected: !model.selected } : model,
    );
    saveProcessedModels();
  }

  function weightBytesFor(result: CalculateResult): number {
    return result.weights ? result.weights.parameterCount * getPrecisionBytes(weightPrecision) : 0;
  }

  function calculateKvAtContext(result: CalculateResult, context: number): number {
    const model = { ...result.model, rawConfig: {} } satisfies NormalizedModelConfig;

    return calculateKvCache({
      model,
      sequenceLength: context,
      batchSize,
      precision,
    }).totalBytes;
  }

  function contextStops(maxContext: number): number[] {
    const stops = 16;
    const step = Math.max(1, Math.floor(maxContext / stops));
    const values = Array.from({ length: stops }, (_, index) => Math.max(1, step * (index + 1)));
    return [...new Set([...values, maxContext])].sort((left, right) => left - right);
  }

  function graphXTicks(maxContext: number): number[] {
    const stops = 4;
    return Array.from({ length: stops + 1 }, (_, index) => Math.round((maxContext / stops) * index));
  }

  function buildGraphScale(series: GraphSeries[]): GraphScale {
    const values = series.flatMap((item) => item.points.map((point) => point.bytes));
    const minValue = values.length > 0 ? Math.min(...values) : 0;
    const maxValue = values.length > 0 ? Math.max(...values) : 1;
    const rawRange = maxValue - minValue;
    const padding = rawRange > 0 ? rawRange * 0.12 : Math.max(maxValue * 0.02, 1);
    const minBytes = Math.max(0, minValue - padding);
    const maxBytes = maxValue + padding;
    const step = (maxBytes - minBytes) / 4;

    return {
      minBytes,
      maxBytes,
      yTicks: Array.from({ length: 5 }, (_, index) => minBytes + step * index).reverse(),
      xTicks: graphXTicks(graphMaxContext),
    };
  }

  function buildGraphSeries(models: ProcessedModel[]): GraphSeries[] {
    return models
      .filter((model): model is ProcessedModel & { result: CalculateResult } => Boolean(model.result))
      .map((model, index) => ({
        id: model.id,
        color: palette[index % palette.length],
        points: contextStops(sequenceLength).map((context) => {
          const kvBytes = calculateKvAtContext(model.result, context);
          const weightBytes = weightBytesFor(model.result);

          return {
            context,
            kvBytes,
            weightBytes,
            bytes: graphBytes(kvBytes, weightBytes),
          };
        }),
      }));
  }

  function hoverContextFromEvent(event: PointerEvent): number {
    const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * 800;
    const clampedX = Math.min(748, Math.max(48, svgX));
    return Math.max(1, Math.round(((clampedX - 48) / 700) * graphMaxContext));
  }

  function handleGraphPointerMove(event: PointerEvent) {
    const context = hoverContextFromEvent(event);
    const items = selectedModels
      .filter((model): model is ProcessedModel & { result: CalculateResult } => Boolean(model.result))
      .map((model, index) => {
        const kvBytes = calculateKvAtContext(model.result, context);
        const weightBytes = weightBytesFor(model.result);
        const bytes = graphBytes(kvBytes, weightBytes);

        return {
          id: model.id,
          color: palette[index % palette.length],
          kvBytes,
          weightBytes,
          bytes,
          y: pointToY(bytes),
        };
      });

    hoveredGraph = {
      context,
      x: pointToX(context),
      items,
    };
  }

  function pointToX(context: number): number {
    return 48 + (context / graphMaxContext) * 700;
  }

  function pointToY(bytes: number): number {
    return 300 - ((bytes - graphScale.minBytes) / (graphScale.maxBytes - graphScale.minBytes)) * 250;
  }

  function linePath(points: GraphPoint[]): string {
    return points
      .map((point, index) => `${index === 0 ? "M" : "L"}${pointToX(point.context).toFixed(2)},${pointToY(point.bytes).toFixed(2)}`)
      .join(" ");
  }

  function modelName(modelId: string): string {
    return modelId.split("/").at(-1) ?? modelId;
  }

  onMount(() => {
    void loadProcessedModels();
  });
</script>

<main class="shell">
  <div class="workspace">
    <header class="brand">
      <h1 aria-label="kvanta"><span>k</span><span>v</span><span>a</span><span>n</span><span>t</span><span>a</span></h1>
      <p>LLM KV-cache and model-weight VRAM calculator.</p>
    </header>

    <section class="picker-panel">
      <form
        class="picker"
        onsubmit={(event) => {
          event.preventDefault();
          void addModel();
        }}
      >
        <label>
          <span>Hugging Face model</span>
          <input
            value={modelInput}
            oninput={(event) => handleModelInput((event.currentTarget as HTMLInputElement).value)}
            placeholder="Search Qwen, Llama, DeepSeek or paste a Hugging Face URL"
            spellcheck="false"
          />
        </label>
        <button type="submit">Add</button>
      </form>
      {#if modelSearchLoading || modelSearchResults.length > 0 || modelSearchError}
        <div class="model-search" aria-live="polite">
          {#if modelSearchLoading}
            <p>Searching Hugging Face models...</p>
          {:else if modelSearchError}
            <p>{modelSearchError}</p>
          {:else if modelSearchResults.length === 0}
            <p>No LLM/VLM models found.</p>
          {:else}
            {#each modelSearchResults as result}
              <button type="button" onclick={() => void addSearchResult(result.id)}>
                <strong>{result.id}</strong>
                <span>{searchResultMeta(result)}</span>
              </button>
            {/each}
          {/if}
        </div>
      {/if}
      {#if pickerError}
        <div class="error slim">{pickerError}</div>
      {/if}
    </section>

    <section class="processed-panel">
      <div class="section-head">
        <h2>Models</h2>
        <span>{selectedModels.length} selected</span>
      </div>

      {#if processedModels.length === 0}
            <p class="empty">Add a Hugging Face model to begin comparing total VRAM footprint growth.</p>
      {:else}
        <div class="model-strip">
          {#each processedModels as model}
            <div class:selected={model.selected} class:error-chip={Boolean(model.error)} class="model-chip">
              <button class="model-select" type="button" onclick={() => toggleModel(model.id)}>
                <strong>{modelName(model.id)}</strong>
                <span>{model.loading ? "loading" : model.error ? "error" : model.id}</span>
              </button>
              <button class="remove" type="button" aria-label={`Remove ${model.id}`} onclick={() => removeModel(model.id)}>
                x
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <section class="config-panel">
      <div class="config-grid">
        <label>
          <span>Context Size</span>
          <input min="1" step="1024" type="number" bind:value={sequenceLength} />
        </label>
        <label>
          <span>Batches</span>
          <input min="1" type="number" bind:value={batchSize} />
        </label>
        <fieldset>
          <legend>KV Precision (bits)</legend>
          <div class="precision-grid">
            {#each kvPrecisionOptions as option}
              <button
                class:active={precision === option.id}
                type="button"
                onclick={() => {
                  precision = option.id;
                }}
              >
                {option.label}
              </button>
            {/each}
          </div>
        </fieldset>
        <fieldset>
          <legend>Weights Quantization (bits)</legend>
          <div class="precision-grid">
            {#each weightPrecisionOptions as option}
              <button
                class:active={weightPrecision === option.id}
                type="button"
                onclick={() => {
                  weightPrecision = option.id;
                }}
              >
                {option.label}
              </button>
            {/each}
          </div>
        </fieldset>
        <fieldset>
          <legend>Graph Scope</legend>
          <div class="mode-switch">
            <button class:active={graphMode === "kv"} type="button" onclick={() => (graphMode = "kv")}>KV only</button>
            <button class:active={graphMode === "total"} type="button" onclick={() => (graphMode = "total")}>KV + weights</button>
          </div>
        </fieldset>
      </div>
    </section>

    <section class="tabs-panel">
      <div class="tabs" role="tablist" aria-label="Model views">
        <button class:active={activeTab === "graph"} type="button" onclick={() => (activeTab = "graph")}>Graph</button>
        <button class:active={activeTab === "architecture"} type="button" onclick={() => (activeTab = "architecture")}>Architecture</button>
      </div>

      {#if activeTab === "graph"}
        <div class="graph-view">
          {#if graphSeries.length === 0}
            <p class="empty">Select at least one processed model to draw the footprint graph.</p>
          {:else}
            <svg
              class="graph"
              viewBox="0 0 800 340"
              role="img"
              aria-label={graphMode === "kv" ? "KV cache growth by context size" : "Total VRAM footprint growth by context size"}
              onpointermove={handleGraphPointerMove}
              onpointerleave={() => (hoveredGraph = null)}
            >
              {#each graphScale.yTicks as tick}
                <line class="grid" x1="48" x2="760" y1={pointToY(tick)} y2={pointToY(tick)} />
                <text x="54" y={pointToY(tick) - 7}>{formatBytes(tick, "gib")}</text>
              {/each}
              {#each graphScale.xTicks as tick}
                <line class="grid" x1={pointToX(tick)} x2={pointToX(tick)} y1="38" y2="300" />
                <text x={pointToX(tick) - 18} y="326">{formatInteger(tick)}</text>
              {/each}
              <line x1="48" x2="760" y1="300" y2="300" />
              <line x1="48" x2="48" y1="38" y2="300" />
              <text x="648" y="34">{graphMode === "kv" ? "kv cache" : "kv + weights"}</text>
              {#each graphSeries as series}
                <path d={linePath(series.points)} stroke={series.color} />
                {#each series.points as point}
                  <circle cx={pointToX(point.context)} cy={pointToY(point.bytes)} r="3" fill={series.color}>
                    <title>{series.id}: {formatBytes(point.bytes, "gib")} shown, {formatBytes(point.kvBytes, "gib")} KV, {formatBytes(point.weightBytes, "gib")} weights at {formatInteger(point.context)} tokens</title>
                  </circle>
                {/each}
              {/each}
              {#if hoveredGraph}
                <line class="hover-line" x1={hoveredGraph.x} x2={hoveredGraph.x} y1="38" y2="300" />
                {#each hoveredGraph.items as item}
                  <circle class="hover-dot" cx={hoveredGraph.x} cy={item.y} r="5" fill={item.color} />
                {/each}
                <g class="tooltip" transform={`translate(${tooltipX(hoveredGraph.x)}, 48)`}>
                  <rect width={tooltipWidth} height={48 + hoveredGraph.items.length * 74} />
                  <text x="12" y="22">ctx {formatInteger(hoveredGraph.context)}</text>
                  {#each hoveredGraph.items as item, index}
                    <circle cx="15" cy={47 + index * 74} r="4" fill={item.color} />
                    <text class="tooltip-name" x="26" y={50 + index * 74}>{compactName(item.id)}</text>
                    <text x="12" y={72 + index * 74}>shown {formatBytes(item.bytes, "gib")}</text>
                    <text x="12" y={89 + index * 74}>kv {formatBytes(item.kvBytes, "gib")}</text>
                    <text x="12" y={106 + index * 74}>weights {formatBytes(item.weightBytes, "gib")}</text>
                  {/each}
                </g>
              {/if}
            </svg>

            <div class="legend">
              {#each graphSeries as series}
                <span><i style={`background:${series.color}`}></i>{modelName(series.id)}</span>
              {/each}
            </div>
          {/if}
        </div>
      {:else}
        <div class="architecture-view">
          {#if selectedModels.length === 0}
            <p class="empty">Select models to inspect architecture fields.</p>
          {:else}
            {#each selectedModels as model}
              {#if model.result}
                <article class="architecture-card">
                  <div class="card-head">
                    <h3>{model.id}</h3>
                    <span>{model.result.model.cacheStrategy.kind}</span>
                  </div>
                  <div class="info-grid">
                    {@render Info("model_type", model.result.model.modelType ?? "unknown")}
                    {@render Info("layers", formatInteger(model.result.model.numHiddenLayers))}
                    {@render Info("hidden", formatInteger(model.result.model.hiddenSize))}
                    {@render Info("attention_heads", formatInteger(model.result.model.numAttentionHeads))}
                    {@render Info("kv_heads", formatInteger(model.result.model.numKeyValueHeads))}
                    {@render Info("head_dim", formatInteger(model.result.model.headDim))}
                    {@render Info("params", model.result.weights ? formatInteger(model.result.weights.parameterCount) : "unknown")}
                    {@render Info("weights", model.result.weights ? formatBytes(weightBytesFor(model.result), "gib") : "unknown")}
                    {@render Info("max_position", model.result.model.maxPositionEmbeddings ? formatInteger(model.result.model.maxPositionEmbeddings) : "unknown")}
                    {@render Info("architecture", model.result.model.architectures[0] ?? "unknown")}
                  </div>
                  {#if model.error}
                    <div class="error">{model.error}</div>
                  {/if}
                </article>
              {/if}
            {/each}
          {/if}
        </div>
      {/if}
    </section>

    <footer class="footer">
      created by <a href="https://github.com/vaclcer" target="_blank" rel="noreferrer">Vaclav Cerny</a>
      <span>/</span>
      <a href="https://github.com/vaclcer/kvanta" target="_blank" rel="noreferrer">GitHub</a>
    </footer>
  </div>
</main>

{#snippet Info(label: string, value: string)}
  <p class="info mono">
    <span>{label}</span>
    <strong>{value}</strong>
  </p>
{/snippet}

<style>
  .shell {
    min-height: 100vh;
    background: var(--paper);
  }

  .workspace {
    width: min(100%, 1180px);
    margin: 0 auto;
    padding: 24px 20px 48px;
  }

  .brand {
    display: grid;
    gap: 9px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--line);
  }

  h1,
  h2,
  h3,
  p {
    margin: 0;
  }

  h1 {
    display: inline-flex;
    align-items: baseline;
    gap: 1px;
    color: transparent;
    font-family: ui-serif, Georgia, "Times New Roman", serif;
    font-size: 38px;
    font-style: italic;
    font-weight: 900;
    line-height: 0.9;
    letter-spacing: 0;
    text-shadow: 2px 0 #111, -1px 0 #2f6df6, 0 2px #d45f00;
  }

  h1 span {
    display: inline-block;
    color: var(--ink);
    transform: translateY(var(--rise, 0)) skewX(var(--skew, -8deg));
  }

  h1 span:nth-child(2n) {
    --rise: -3px;
    --skew: 6deg;
  }

  h1 span:nth-child(3n) {
    --rise: 2px;
    --skew: -13deg;
  }

  .brand p {
    color: rgb(17 17 17 / 52%);
    font-size: 13px;
    font-weight: 650;
    line-height: 1.35;
  }

  h2 {
    font-size: 14px;
    font-weight: 750;
    line-height: 1.2;
  }

  h3 {
    min-width: 0;
    overflow: hidden;
    font-size: 15px;
    line-height: 1.2;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .picker-panel,
  .processed-panel,
  .config-panel,
  .tabs-panel {
    margin-top: 16px;
    border: 1px solid var(--line);
    background: var(--panel);
  }

  .picker-panel,
  .processed-panel,
  .config-panel {
    padding: 16px;
  }

  .config-panel {
    padding: 12px;
  }

  .picker {
    display: grid;
    gap: 12px;
  }

  .model-search {
    display: grid;
    max-height: 312px;
    margin-top: 12px;
    overflow-y: auto;
    border: 1px solid var(--line);
    background: var(--paper);
  }

  .model-search p {
    padding: 11px 12px;
    color: rgb(17 17 17 / 55%);
    font-size: 13px;
    font-weight: 700;
  }

  .model-search button {
    display: grid;
    gap: 3px;
    min-width: 0;
    border: 0;
    border-bottom: 1px solid var(--line);
    background: transparent;
    padding: 10px 12px;
    color: var(--ink);
    text-align: left;
  }

  .model-search button:last-child {
    border-bottom: 0;
  }

  .model-search button:hover {
    background: var(--ink);
    color: white;
  }

  .model-search strong,
  .model-search span {
    overflow: hidden;
    overflow-wrap: anywhere;
  }

  .model-search span {
    color: rgb(17 17 17 / 55%);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 11px;
  }

  .model-search button:hover span {
    color: rgb(255 255 255 / 68%);
  }

  label,
  fieldset {
    display: grid;
    gap: 8px;
    min-width: 0;
  }

  fieldset {
    padding: 0;
    border: 0;
  }

  label span,
  legend {
    font-size: 14px;
    font-weight: 750;
    line-height: 1.2;
  }

  legend {
    margin-bottom: 8px;
  }

  input,
  button {
    border-radius: 0;
    font: inherit;
  }

  input,
  .picker > button,
  .precision-grid button,
  .mode-switch button,
  .tabs button {
    min-height: 44px;
    border: 1px solid rgb(17 17 17 / 15%);
    background: var(--paper);
    color: var(--ink);
    outline: none;
  }

  .config-panel input,
  .config-panel .precision-grid button,
  .config-panel .mode-switch button {
    min-height: 36px;
  }

  input {
    width: 100%;
    min-width: 0;
    padding: 0 12px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 14px;
  }

  input:focus {
    border-color: var(--ink);
  }

  button {
    cursor: pointer;
  }

  .picker > button,
  .precision-grid button,
  .mode-switch button,
  .tabs button {
    padding: 0 14px;
    font-weight: 750;
  }

  .config-panel .precision-grid button,
  .config-panel .mode-switch button {
    padding: 0 10px;
  }

  .picker > button:hover,
  .precision-grid button:hover,
  .precision-grid button.active,
  .mode-switch button:hover,
  .mode-switch button.active,
  .tabs button:hover,
  .tabs button.active {
    border-color: var(--ink);
    background: var(--ink);
    color: white;
  }

  .section-head,
  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .section-head span,
  .card-head span {
    flex: none;
    color: rgb(17 17 17 / 50%);
    font-size: 12px;
    font-weight: 700;
  }

  .model-strip,
  .legend {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-top: 12px;
  }

  .model-chip {
    display: flex;
    flex: 0 0 min(300px, 70vw);
    min-width: 0;
    border: 1px solid rgb(17 17 17 / 12%);
    background: var(--paper);
  }

  .model-chip.selected {
    border-color: var(--ink);
    box-shadow: 4px 4px 0 var(--ink);
  }

  .model-chip.error-chip {
    border-color: rgb(255 77 77 / 45%);
    background: #fff0f0;
  }

  .model-select {
    display: grid;
    flex: 1;
    min-width: 0;
    gap: 2px;
    border: 0;
    background: transparent;
    padding: 9px 10px;
    text-align: left;
  }

  .model-select strong,
  .model-select span {
    overflow: hidden;
    overflow-wrap: anywhere;
  }

  .model-select span {
    color: rgb(17 17 17 / 55%);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 11px;
  }

  .remove {
    width: 32px;
    border: 0;
    border-left: 1px solid var(--line);
    background: transparent;
    color: rgb(17 17 17 / 55%);
  }

  .remove:hover {
    background: var(--ink);
    color: white;
  }

  .precision-grid,
  .mode-switch,
  .info-grid {
    display: grid;
    gap: 12px;
  }

  .config-grid {
    display: grid;
    grid-template-columns: 150px 92px 214px 250px 186px;
    align-items: flex-start;
    gap: 18px 22px;
  }

  .precision-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
  }

  .mode-switch {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  @media (max-width: 1040px) {
    .config-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 640px) {
    .config-grid {
      grid-template-columns: 1fr;
    }
  }

  .tabs {
    display: flex;
    gap: 8px;
    border-bottom: 1px solid var(--line);
    padding: 12px;
  }

  .tabs button {
    min-height: 36px;
  }

  .graph-view,
  .architecture-view {
    padding: 16px;
  }

  .graph {
    display: block;
    width: 100%;
    min-height: 320px;
    background: var(--paper);
    border: 1px solid var(--line);
  }

  .graph line {
    stroke: rgb(17 17 17 / 25%);
    stroke-width: 1;
  }

  .graph line.grid {
    stroke: rgb(17 17 17 / 8%);
  }

  .graph .hover-line {
    stroke: rgb(17 17 17 / 45%);
    stroke-dasharray: 4 4;
  }

  .graph .hover-dot {
    stroke: white;
    stroke-width: 2;
  }

  .graph path {
    fill: none;
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .graph text {
    fill: rgb(17 17 17 / 55%);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12px;
  }

  .tooltip {
    pointer-events: none;
  }

  .tooltip rect {
    fill: rgb(255 255 255 / 96%);
    stroke: var(--ink);
    stroke-width: 1;
  }

  .tooltip text {
    fill: var(--ink);
    font-size: 11px;
  }

  .tooltip .tooltip-name {
    font-weight: 700;
  }

  .legend {
    flex-wrap: wrap;
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--line);
    background: var(--paper);
    padding: 7px 10px;
    font-size: 13px;
  }

  .legend i {
    width: 10px;
    height: 10px;
  }

  .architecture-view {
    display: grid;
    gap: 14px;
  }

  .architecture-card {
    display: grid;
    gap: 12px;
    border: 1px solid var(--line);
    background: var(--paper);
    padding: 14px;
  }

  .info-grid {
    grid-template-columns: 1fr;
  }

  .info {
    display: flex;
    gap: 8px;
    min-width: 0;
    border: 1px solid var(--line);
    background: var(--panel);
    padding: 10px 12px;
    font-size: 14px;
  }

  .info span {
    flex: none;
    color: rgb(17 17 17 / 40%);
  }

  .info strong {
    min-width: 0;
    overflow: hidden;
    color: var(--ink);
    font-weight: 500;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .empty {
    color: rgb(17 17 17 / 55%);
    font-size: 14px;
  }

  .error {
    border: 1px solid rgb(255 77 77 / 40%);
    background: #fff0f0;
    color: #8a1c1c;
    padding: 14px;
    font-size: 14px;
  }

  .slim {
    margin-top: 10px;
    padding: 10px;
  }

  .mono {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }

  .footer {
    margin-top: 18px;
    color: rgb(17 17 17 / 45%);
    font-size: 12px;
  }

  .footer a {
    color: inherit;
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  .footer a:hover {
    color: var(--ink);
  }

  @media (min-width: 720px) {
    .workspace {
      padding-inline: 32px;
    }

    .picker {
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
    }

    .config-grid > label:first-child {
      flex-basis: 170px;
    }

    .config-grid > label:nth-child(2) {
      flex-basis: 105px;
    }

    .info-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (min-width: 1040px) {
    .workspace {
      padding-inline: 40px;
    }

    .info-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }

  @media (min-width: 1280px) {
    .workspace {
      width: min(100%, 1320px);
    }

    .config-grid {
      gap: 18px 46px;
    }

    .config-grid > fieldset {
      flex-basis: 250px;
    }
  }
</style>
