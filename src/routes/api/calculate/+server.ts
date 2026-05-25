import { json } from "@sveltejs/kit";
import { calculateKvCache } from "$lib/calculation/kvCache";
import { estimateWeights } from "$lib/calculation/weights";
import { configSourceUrl, fetchModelConfig, fetchModelInfo } from "$lib/huggingface/client";
import { normalizeConfig } from "$lib/models/normalizeConfig";
import type { CalculateRequest } from "$lib/models/types";
import type { RequestHandler } from "./$types";

function asPositiveInteger(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : fallback;
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = (await request.json()) as CalculateRequest;
    const modelId = body.modelId?.trim();

    if (!modelId && !body.config) {
      return json({ error: "Provide either modelId or raw Hugging Face config." }, { status: 400 });
    }

    const rawConfig = body.config ?? (await fetchModelConfig(modelId as string));
    const resolvedModelId = modelId ?? "manual-config";
    const model = normalizeConfig(rawConfig, {
      modelId: resolvedModelId,
      sourceUrl: modelId ? configSourceUrl(modelId) : "manual-config",
    });
    const result = calculateKvCache({
      model,
      sequenceLength: asPositiveInteger(body.sequenceLength, 4096),
      batchSize: asPositiveInteger(body.batchSize, 1),
      precision: body.precision ?? "float16",
    });

    if (modelId) {
      const modelInfo = await fetchModelInfo(modelId);

      if (modelInfo.safetensors?.total) {
        result.weights = estimateWeights(modelInfo.safetensors.total, body.weightPrecision ?? body.precision ?? "float16");
      } else {
        result.warnings = [
          ...result.warnings,
          "Hugging Face did not expose safetensors parameter metadata, so model weight footprint is unavailable.",
        ];
      }
    }

    return json({ result });
  } catch (error) {
    return json(
      { error: error instanceof Error ? error.message : "Calculation failed." },
      { status: 400 },
    );
  }
};