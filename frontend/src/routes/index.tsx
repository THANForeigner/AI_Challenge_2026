import { createFileRoute } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";

import { Toaster } from "@/components/ui/sonner";
import { SearchSidebar } from "@/components/search/SearchSidebar";
import { ResultGrid } from "@/components/search/ResultGrid";
import { DetailModal } from "@/components/search/DetailModal";
import { searchKeyframes } from "@/lib/pltn.functions";
import type {
  KeyframeResult,
  SearchMode,
  SearchModel,
  SearchParams,
  SearchResponse,
  TemporalEvent,
} from "@/lib/pltn.types";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TongFC Search — AI Keyframe Search" },
      {
        name: "description",
        content:
          "Search video keyframes by text, image, OCR, ASR, or temporal events with the TongFC engine.",
      },
      { property: "og:title", content: "TongFC Search — AI Keyframe Search" },
      {
        property: "og:description",
        content: "Search video keyframes by text, image, OCR, ASR, or temporal events.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Something went wrong while searching.";
}

function buildParams(fields: {
  mode: SearchMode;
  query: string;
  imageBase64: string | undefined;
  ocrFilter: string;
  asrFilter: string;
  temporalEvents: TemporalEvent[];
  topK: number;
  model: SearchModel;
  apiBaseUrl: string;
}): SearchParams {
  return {
    mode: fields.mode,
    topK: fields.topK,
    model: fields.model,
    ...(fields.query ? { query: fields.query } : {}),
    ...(fields.imageBase64 ? { imageBase64: fields.imageBase64 } : {}),
    ...(fields.ocrFilter ? { ocrFilter: fields.ocrFilter } : {}),
    ...(fields.asrFilter ? { asrFilter: fields.asrFilter } : {}),
    ...(fields.temporalEvents.length > 0 ? { temporalEvents: fields.temporalEvents } : {}),
    ...(fields.apiBaseUrl ? { apiBaseUrl: fields.apiBaseUrl } : {}),
  };
}

function Index() {
  const [mode, setMode] = useState<SearchMode>("text");
  const [query, setQuery] = useState("");
  const [imageBase64, setImageBase64] = useState<string | undefined>(undefined);
  const [ocrFilter, setOcrFilter] = useState("");
  const [asrFilter, setAsrFilter] = useState("");
  const [temporalEvents, setTemporalEvents] = useState<TemporalEvent[]>([]);
  const [model, setModel] = useState<SearchModel>("beit3");
  const [topK, setTopK] = useState(100);
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [detailIndex, setDetailIndex] = useState<number | null>(null);
  const [density, setDensity] = useState(4);

  const searchMutation = useMutation({
    mutationFn: (params: SearchParams) => searchKeyframes({ data: params }),
  });

  const search: SearchResponse | undefined = searchMutation.data;
  const results = search?.results ?? [];

  function runSearch() {
    searchMutation.mutate(
      buildParams({
        mode,
        query,
        imageBase64,
        ocrFilter,
        asrFilter,
        temporalEvents,
        topK,
        model,
        apiBaseUrl,
      }),
    );
  }

  function handleSimilar(result: KeyframeResult) {
    setMode("image");
    setImageBase64(result.imageUrl);
    searchMutation.mutate(
      buildParams({
        mode: "image",
        query: "",
        imageBase64: result.imageUrl,
        ocrFilter: "",
        asrFilter: "",
        temporalEvents: [],
        topK,
        model,
        apiBaseUrl,
      }),
    );
  }

  function handleClear() {
    searchMutation.reset();
    setDetailIndex(null);
  }

  function goDetail(delta: number) {
    if (detailIndex == null || results.length === 0) return;
    setDetailIndex((detailIndex + delta + results.length) % results.length);
  }

  async function handleCopySubmit(result: KeyframeResult) {
    const line = `${result.videoId}, ${result.frameIdx}`;
    try {
      await navigator.clipboard.writeText(line);
      toast.success("Copied to clipboard", { description: line });
    } catch {
      toast.error("Could not copy to clipboard");
    }
  }

  const current =
    detailIndex != null && detailIndex >= 0 && detailIndex < results.length
      ? results[detailIndex]!
      : null;

  return (
    <div className="flex h-screen overflow-hidden">
      <SearchSidebar
        mode={mode}
        setMode={setMode}
        query={query}
        setQuery={setQuery}
        imageBase64={imageBase64}
        setImageBase64={setImageBase64}
        ocrFilter={ocrFilter}
        setOcrFilter={setOcrFilter}
        asrFilter={asrFilter}
        setAsrFilter={setAsrFilter}
        temporalEvents={temporalEvents}
        setTemporalEvents={setTemporalEvents}
        model={model}
        setModel={setModel}
        topK={topK}
        setTopK={setTopK}
        onSearch={runSearch}
        loading={searchMutation.isPending}
        apiBaseUrl={apiBaseUrl}
        setApiBaseUrl={setApiBaseUrl}
      />

      <main className="min-w-0 flex-1 overflow-y-auto p-4 lg:p-6">
        <ResultGrid
          search={search}
          onView={(result) => setDetailIndex(results.findIndex((r) => r.id === result.id))}
          onSimilar={handleSimilar}
          loading={searchMutation.isPending}
          error={searchMutation.isError ? getErrorMessage(searchMutation.error) : null}
          onClear={handleClear}
          density={density}
          setDensity={setDensity}
        />
      </main>

      <DetailModal
        result={current}
        open={current != null}
        onOpenChange={(open) => {
          if (!open) setDetailIndex(null);
        }}
        onPrev={() => goDetail(-1)}
        onNext={() => goDetail(1)}
        canPrev={results.length > 1}
        canNext={results.length > 1}
        onCopySubmit={handleCopySubmit}
        onSimilar={handleSimilar}
      />

      <Toaster theme="dark" position="bottom-right" />
    </div>
  );
}
