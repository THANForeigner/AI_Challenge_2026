export type SearchMode = "text" | "image" | "ocr" | "asr" | "temporal";

export type SearchModel =
  | "beit3"
  | "clip"
  | "blip2"
  | "minigpt4"
  | "llava"
  | string;

export interface TemporalEvent {
  id: string;
  text: string;
  order: number;
}

export interface SearchParams {
  mode: SearchMode;
  query?: string;
  imageBase64?: string;
  ocrFilter?: string;
  asrFilter?: string;
  temporalEvents?: TemporalEvent[];
  topK: number;
  model: SearchModel;
}

export interface KeyframeResult {
  id: string;
  videoId: string;
  frameIdx: number;
  score: number;
  rank: number;
  imageUrl: string | undefined;
  timestamp: number | undefined;
  ocr: string | undefined;
  asr: string | undefined;
  source: "api" | "demo";
}

export interface SearchResponse {
  results: KeyframeResult[];
  total: number;
  elapsedMs: number;
  mode: SearchMode;
  source: "api" | "demo";
}

export interface RawApiItem {
  video_id?: string;
  videoId?: string;
  video_name?: string;
  video?: string;
  frame?: number;
  frame_idx?: number;
  frameIdx?: number;
  idx?: number;
  score?: number;
  similarity?: number;
  rank?: number;
  path?: string;
  url?: string;
  image?: string;
  image_url?: string;
  timestamp?: number;
  time?: number;
  ocr?: string;
  asr?: string;
  ocr_text?: string;
  asr_text?: string;
}

export interface RawApiResponse {
  results?: RawApiItem[];
  data?: RawApiItem[];
  items?: RawApiItem[];
  keyframes?: RawApiItem[];
  total?: number;
  elapsed?: number;
  elapsed_time?: number;
  query?: string;
}
