import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { makeDemoResponse, normalizeApiResponse } from "./pltn-normalize";
import type { SearchParams, SearchResponse } from "./pltn.types";

const SearchPayloadSchema = z.object({
  mode: z.enum(["text", "image", "temporal"]),
  query: z.string().optional(),
  imageBase64: z.string().optional(),
  ocrFilter: z.string().optional(),
  asrFilter: z.string().optional(),
  temporalEvents: z
    .array(
      z.object({
        id: z.string(),
        text: z.string(),
        order: z.number(),
      }),
    )
    .optional(),
  topK: z.number().min(1).max(500),
  model: z.string(),
  apiBaseUrl: z.string().optional(),
});

function getApiBaseUrl(): string | undefined {
  return process.env["PLTN_API_BASE_URL"] || process.env["PIKA_API_BASE_URL"];
}

function buildApiBody(params: SearchParams): unknown {
  const base = {
    mode: params.mode,
    model: params.model,
    top_k: params.topK,
  };

  switch (params.mode) {
    case "image":
      return {
        ...base,
        image_base64: params.imageBase64,
      };
    case "temporal":
      return {
        ...base,
        query: params.temporalEvents?.map((e) => e.text) ?? [],
      };
    default:
      return {
        ...base,
        query: params.query,
        ...(params.ocrFilter ? { ocr_filter: params.ocrFilter } : {}),
        ...(params.asrFilter ? { asr_filter: params.asrFilter } : {}),
      };
  }
}

export const searchKeyframes = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => SearchPayloadSchema.parse(input))
  .handler(async ({ data }): Promise<SearchResponse> => {
    const start = performance.now();
    const baseUrl = data.apiBaseUrl?.trim() || getApiBaseUrl();

    if (!baseUrl) {
      return makeDemoResponse(data.mode, data.query, data.topK);
    }

    const endpoint = `${baseUrl.replace(/\/$/, "")}/search`;
    const body = buildApiBody(data as SearchParams);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}: ${await response.text()}`);
      }

      const raw = (await response.json()) as Record<string, unknown>;
      return normalizeApiResponse(raw, data.mode, performance.now() - start);
    } catch (error) {
      console.error("PLTN search failed, falling back to demo:", error);
      return makeDemoResponse(data.mode, data.query, data.topK);
    }
  });
