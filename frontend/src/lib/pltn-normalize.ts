import type {
  KeyframeResult,
  RawApiItem,
  RawApiResponse,
  SearchMode,
  SearchResponse,
} from "./pltn.types";

function pickVideoId(raw: RawApiItem): string {
  return raw.video_id ?? raw.videoId ?? raw.video_name ?? raw.video ?? "unknown";
}

function pickFrameIdx(raw: RawApiItem): number {
  if (typeof raw.frame === "number") return raw.frame;
  if (typeof raw.frame_idx === "number") return raw.frame_idx;
  if (typeof raw.frameIdx === "number") return raw.frameIdx;
  if (typeof raw.idx === "number") return raw.idx;
  return 0;
}

function pickScore(raw: RawApiItem): number {
  if (typeof raw.score === "number") return raw.score;
  if (typeof raw.similarity === "number") return raw.similarity;
  return 0;
}

function pickImageUrl(raw: RawApiItem, videoId: string, frameIdx: number): string | undefined {
  const candidates = [raw.image_url, raw.url, raw.image, raw.path].filter(
    (x): x is string => typeof x === "string" && x.length > 0,
  );
  if (candidates.length > 0) return candidates[0];
  return `https://placehold.co/320x180/1a2035/9ca3af?text=${encodeURIComponent(
    `${videoId}\nframe ${frameIdx}`,
  )}`;
}

function pickTimestamp(raw: RawApiItem): number | undefined {
  if (typeof raw.timestamp === "number") return raw.timestamp;
  if (typeof raw.time === "number") return raw.time;
  return undefined;
}

function pickOcr(raw: RawApiItem): string | undefined {
  return raw.ocr ?? raw.ocr_text ?? undefined;
}

function pickAsr(raw: RawApiItem): string | undefined {
  return raw.asr ?? raw.asr_text ?? undefined;
}

export function normalizeApiResponse(
  raw: RawApiResponse | RawApiItem[],
  mode: SearchMode,
  elapsedMs: number,
): SearchResponse {
  const source: "api" | "demo" = "api";

  let payload: RawApiResponse;
  if (Array.isArray(raw)) {
    payload = { results: raw };
  } else {
    payload = raw ?? {};
  }

  const rawItems: RawApiItem[] =
    payload.results ?? payload.data ?? payload.items ?? payload.keyframes ?? [];

  const results: KeyframeResult[] = rawItems
    .map((item, index) => {
      const videoId = pickVideoId(item);
      const frameIdx = pickFrameIdx(item);
      const score = pickScore(item);
      return {
        id: `${videoId}-${frameIdx}-${index}`,
        videoId,
        frameIdx,
        score,
        rank: item.rank ?? index + 1,
        imageUrl: pickImageUrl(item, videoId, frameIdx),
        timestamp: pickTimestamp(item),
        ocr: pickOcr(item),
        asr: pickAsr(item),
        source,
      };
    })
    .filter((r) => r.videoId !== "unknown" || r.frameIdx !== 0);

  return {
    results,
    total: payload.total ?? results.length,
    elapsedMs: payload.elapsed ?? payload.elapsed_time ?? elapsedMs,
    mode,
    source,
  };
}

const DEMO_IMAGES = [
  "fireworks",
  "concert",
  "march",
  "rice-field",
  "speech",
  "sport",
  "festival",
  "classroom",
  "market",
  "traffic",
];

export function makeDemoResponse(mode: SearchMode, query = "", topK = 100): SearchResponse {
  const results: KeyframeResult[] = Array.from({ length: Math.min(topK, 60) }).map((_, i) => {
    const videoId = `L${String((i % 5) + 1).padStart(2, "0")}_V${String(
      Math.floor(i / 5) + 1,
    ).padStart(3, "0")}`;
    const frameIdx = Math.floor(Math.random() * 5000) + 100;
    const imageKeyword = DEMO_IMAGES[i % DEMO_IMAGES.length];
    return {
      id: `${videoId}-${frameIdx}-${i}`,
      videoId,
      frameIdx,
      score: 1 - i * 0.01 - Math.random() * 0.05,
      rank: i + 1,
      imageUrl: `https://placehold.co/320x180/1a2035/9ca3af?text=${encodeURIComponent(
        `${imageKeyword}\n${videoId}\nframe ${frameIdx}`,
      )}`,
      timestamp: frameIdx / 25,
      ocr: `${imageKeyword} scene`,
      asr: `audio transcript ${i + 1}`,
      source: "demo",
    };
  });

  return {
    results,
    total: results.length,
    elapsedMs: 420 + Math.floor(Math.random() * 300),
    mode,
    source: "demo",
  };
}
