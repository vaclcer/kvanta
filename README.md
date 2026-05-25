# kvanta

Sharp little VRAM calculator for open LLMs. ⚡

Live app: [https://kvanta.vcerny.cz](https://kvanta.vcerny.cz)  
Source: [github.com/vaclcer/kvanta](https://github.com/vaclcer/kvanta)

kvanta fetches public Hugging Face model configs and safetensors metadata, then calculates KV-cache memory and estimated model-weight footprint from context size, batch count, precision, and quantization.

## Notes

- Exact adapters for standard decoder-only, GQA/MQA, sliding-window, GLM MoE DSA, DeepSeek V4, and Qwen3.5 hybrid cache layouts.
- Model weights are estimated from Hugging Face `safetensors.total` metadata.
- Runtime VRAM can still drift by inference engine because allocators, kernels, paged attention, and activation buffers all have opinions.

## Development

```bash
npm install
npm run dev
```

Useful checks:

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## Deployment

The app is deployed on Vercel. 🚀

`kvanta.vcerny.cz` is a custom domain pointed to Vercel with a `CNAME` record.

For gated or private Hugging Face models, set this environment variable in Vercel:

```txt
HUGGINGFACE_TOKEN=...
```
