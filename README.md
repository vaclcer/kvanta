# kvanta

KV-cache and model-weight VRAM calculator for open LLMs.

Live app: [https://kvanta.vcerny.cz](https://kvanta.vcerny.cz)

kvanta fetches public Hugging Face model configs and safetensors metadata, then calculates KV-cache memory and estimated model-weight footprint from context size, batch count, precision, and quantization.

## Notes

- Standard decoder-only, GQA/MQA, sliding-window, GLM MoE DSA, DeepSeek V4, and Qwen3.5 hybrid cache layouts are handled explicitly.
- Model weights are estimated from Hugging Face `safetensors.total` parameter metadata.
- Runtime VRAM can still differ by inference engine because of allocator reserve, temporary activations, kernels, paged attention, and serving framework choices.

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

The app is deployed on Vercel.

`kvanta.vcerny.cz` is a custom domain pointed to Vercel with a `CNAME` record.

For gated or private Hugging Face models, set this environment variable in Vercel:

```txt
HUGGINGFACE_TOKEN=...
```

## HTTPS

Vercel manages HTTPS automatically for custom domains. After the `CNAME` is configured and Vercel verifies the domain, it provisions and renews the TLS certificate for `https://kvanta.vcerny.cz`.

If HTTPS is not active yet, wait for DNS propagation and check the domain status in Vercel Project Settings -> Domains.
