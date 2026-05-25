import { json } from "@sveltejs/kit";
import { searchHuggingFaceModels } from "$lib/huggingface/client";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q")?.trim() ?? "";

    if (query.length < 2) {
      return json({ results: [] });
    }

    const results = await searchHuggingFaceModels(query);
    return json({ results });
  } catch (error) {
    return json(
      { error: error instanceof Error ? error.message : "Model search failed." },
      { status: 400 },
    );
  }
};