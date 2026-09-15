import { ImagePlus, Search, Settings } from "lucide-react";
import { Logo } from "./Logo";
import { ModeTabs } from "./ModeTabs";
import { TemporalEditor } from "./TemporalEditor";
import { ApiSettings } from "./ApiSettings";
import type { SearchMode, SearchModel, TemporalEvent } from "@/lib/pltn.types";

interface SearchSidebarProps {
  mode: SearchMode;
  setMode: (mode: SearchMode) => void;
  query: string;
  setQuery: (value: string) => void;
  imageBase64: string | undefined;
  setImageBase64: (value: string | undefined) => void;
  ocrFilter: string;
  setOcrFilter: (value: string) => void;
  asrFilter: string;
  setAsrFilter: (value: string) => void;
  temporalEvents: TemporalEvent[];
  setTemporalEvents: (events: TemporalEvent[]) => void;
  model: SearchModel;
  setModel: (model: SearchModel) => void;
  topK: number;
  setTopK: (value: number) => void;
  onSearch: () => void;
  loading: boolean;
  apiBaseUrl: string;
  setApiBaseUrl: (value: string) => void;
}

const MODELS: SearchModel[] = ["beit3", "clip", "blip2", "minigpt4", "llava"];

export function SearchSidebar(props: SearchSidebarProps) {
  const {
    mode,
    setMode,
    query,
    setQuery,
    imageBase64,
    setImageBase64,
    ocrFilter,
    setOcrFilter,
    asrFilter,
    setAsrFilter,
    temporalEvents,
    setTemporalEvents,
    model,
    setModel,
    topK,
    setTopK,
    onSearch,
    loading,
    apiBaseUrl,
    setApiBaseUrl,
  } = props;

  function handleImageUpload(file: File | undefined) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        setImageBase64(result);
      }
    };
    reader.readAsDataURL(file);
  }

  return (
    <aside className="flex h-full w-full flex-col gap-4 overflow-y-auto border-r border-border bg-sidebar p-4 lg:w-72 lg:shrink-0">
      <Logo />

      <ModeTabs value={mode} onChange={setMode} />

      <div className="space-y-3 rounded-lg border border-border bg-panel p-3">
        {mode === "image" ? (
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Upload image
            </p>
            <label
              htmlFor="image-upload"
              className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border bg-background px-4 py-8 text-center transition-colors hover:border-primary hover:bg-secondary"
            >
              {imageBase64 ? (
                <img src={imageBase64} alt="Query" className="max-h-32 rounded-md object-contain" />
              ) : (
                <>
                  <ImagePlus className="h-8 w-8 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Drop or click to upload</span>
                </>
              )}
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleImageUpload(e.target.files?.[0])}
              />
            </label>
            {imageBase64 && (
              <button
                type="button"
                onClick={() => setImageBase64(undefined)}
                className="text-xs text-destructive hover:underline"
              >
                Remove image
              </button>
            )}
          </div>
        ) : mode === "temporal" ? (
          <TemporalEditor events={temporalEvents} onChange={setTemporalEvents} />
        ) : (
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Text query
            </p>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  onSearch();
                }
              }}
              placeholder="Enter query..."
              rows={4}
              className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none ring-ring focus:ring-1"
            />
          </div>
        )}
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-panel p-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Select Model
        </p>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none ring-ring focus:ring-1"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-panel p-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Filter Panel
        </p>
        <input
          type="text"
          value={ocrFilter}
          onChange={(e) => setOcrFilter(e.target.value)}
          placeholder="OCR Filter..."
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none ring-ring focus:ring-1"
        />
        <input
          type="text"
          value={asrFilter}
          onChange={(e) => setAsrFilter(e.target.value)}
          placeholder="ASR Filter..."
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none ring-ring focus:ring-1"
        />
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-panel p-3">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Top K
          </p>
          <span className="text-xs font-mono text-neon-light-pink">{topK}</span>
        </div>
        <input
          type="range"
          min={10}
          max={500}
          step={10}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="w-full accent-primary"
        />
      </div>

      <button
        type="button"
        onClick={onSearch}
        disabled={loading}
        className="mt-auto flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-bold text-primary-foreground shadow-[0_0_16px_oklch(0.72_0.22_345/40%)] transition-all hover:bg-primary/90 hover:shadow-[0_0_24px_oklch(0.72_0.22_345/55%)] disabled:opacity-60"
      >
        <Search className="h-4 w-4" />
        {loading ? "Searching..." : "Search"}
      </button>

      <ApiSettings baseUrl={apiBaseUrl} onChange={setApiBaseUrl} />
    </aside>
  );
}
