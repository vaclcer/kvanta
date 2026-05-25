kvanta is a full-stack SvelteKit calculator for LLM VRAM footprint. It fetches Hugging Face `config.json` and safetensors metadata server-side, normalizes layer metadata, and calculates KV-cache bytes plus estimated model-weight bytes from user-selected context, batch size, KV precision, and weight quantization.

## Scope

The first version calculates KV-cache VRAM for decoder-only LLMs with standard attention, GQA/MQA, sliding-window attention, GLM MoE DSA's Hugging Face Transformers cache layout, and DeepSeek V4's hybrid sparse cache layout. It is exact when the Hugging Face config contains the fields needed for the formula.

It does not claim exact full runtime VRAM across inference engines. Runtime totals depend on model weights, temporary activations, allocator behavior, kernels, paged-attention layout, and serving framework choices. Ambiguous architectures such as MLA/compressed-cache models return explicit unsupported diagnostics until an adapter can calculate them from public config fields.

Model weight footprint is estimated from Hugging Face `safetensors.total` parameter metadata multiplied by the selected weight quantization byte width. This captures parameter storage, not serving-engine overhead, optimizer state, allocator reserve, or temporary activation buffers.

## Formula

For each attention layer:

```txt
layer_bytes = batch * tokens * num_key_value_heads * head_dim * 2 * kv_precision_bytes
```

Sliding-window layers use `min(sequence_length, sliding_window)` for `tokens`. The factor of `2` accounts for key and value tensors.

For `glm_moe_dsa` models such as GLM-5.1, kvanta uses the Transformers implementation layout: expanded attention keys `[batch, heads, tokens, qk_head_dim]`, expanded values `[batch, heads, tokens, v_head_dim]`, and the per-layer DSA indexer key cache `[batch, tokens, index_head_dim]`.

For `deepseek_v4` models such as DeepSeek-V4-Pro, kvanta uses the official hybrid sparse layout exposed by the config: sliding latent KV from `sliding_window * head_dim`, compressed latent KV from `floor(tokens / compress_ratio) * head_dim`, and FP16 indexer cache for ratio-4 layers from `floor(tokens / 4) * index_head_dim * 2`.

## Development

First, run the development server:

```bash
npm run dev
```

Open the local URL printed by Vite, usually [http://localhost:5173](http://localhost:5173).

Useful checks:

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## Hugging Face Access

Public model configs work without credentials. For gated or private models, set `HUGGINGFACE_TOKEN` in the server environment.

## Model Input

Paste a Hugging Face model URL or enter an `owner/model-name` ID. Models that calculate successfully are saved in browser storage as recent entries. Unsupported or incomplete configs return a clear error instead of an estimate.

## Deployment

Use a SvelteKit-supported server host such as Vercel, Netlify, Cloudflare, or Node. GitHub Pages alone cannot run the backend API routes.

## Roadmap

- Add exact weight-storage calculation from safetensors metadata.
- Add generated registry refresh scripts for verified model configs.
- Add model-specific adapters for DeepSeek MLA/compressed-cache layouts.
- Add selectable serving-engine profiles for non-exact runtime overhead reporting.
