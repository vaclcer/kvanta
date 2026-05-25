import { json } from "@sveltejs/kit";
import { configSourceUrl, fetchModelConfig } from "$lib/huggingface/client";
import { normalizeConfig } from "$lib/models/normalizeConfig";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ params }) => {
  try {
    const modelId = params.modelId;
    const rawConfig = await fetchModelConfig(modelId);
    const normalized = normalizeConfig(rawConfig, {
      modelId,
      sourceUrl: configSourceUrl(modelId),
    });

    return json({ config: normalized });
  } catch (error) {
    return json(
      { error: error instanceof Error ? error.message : "Unable to load model config." },
      { status: 400 },
    );
  }
};