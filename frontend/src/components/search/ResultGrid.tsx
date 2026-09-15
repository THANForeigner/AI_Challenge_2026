import { LayoutGrid, SearchX, Trash2 } from "lucide-react";
import { KeyframeCard } from "./KeyframeCard";
import { Skeleton } from "@/components/ui/skeleton";
import type { KeyframeResult, SearchResponse } from "@/lib/pltn.types";

const DENSITY_OPTIONS = [2, 3, 4, 5, 6, 7];

interface ResultGridProps {
  search: SearchResponse | undefined;
  onView: (result: KeyframeResult) => void;
  onSimilar: (result: KeyframeResult) => void;
  loading: boolean;
  error: string | null;
  onClear: () => void;
  density: number;
  setDensity: (value: number) => void;
}

export function ResultGrid({
  search,
  onView,
  onSimilar,
  loading,
  error,
  onClear,
  density,
  setDensity,
}: ResultGridProps) {
  const results = search?.results ?? [];

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-panel px-3 py-2">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            <span className="font-mono font-bold text-foreground">{results.length}</span> results
          </span>
          {search && <span className="font-mono">{(search.elapsedMs / 1000).toFixed(2)}s</span>}
          {search && (
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                search.source === "demo"
                  ? "bg-neon-amber/90 text-black"
                  : "bg-neon-pink/20 text-neon-pink"
              }`}
            >
              {search.source}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-1 rounded-md border border-border bg-background p-1 sm:flex">
            {DENSITY_OPTIONS.map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDensity(d)}
                title={`${d} columns`}
                className={`grid h-6 w-6 place-items-center rounded ${density === d ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary hover:text-foreground"}`}
              >
                <LayoutGrid className="h-3.5 w-3.5" />
              </button>
            ))}
          </div>
          {results.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              className="flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1.5 text-[11px] font-semibold text-destructive hover:bg-destructive/20"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading ? (
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${density}, minmax(0, 1fr))` }}
        >
          {Array.from({ length: density * 6 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="aspect-video w-full rounded-lg" />
              <Skeleton className="h-3 w-2/3" />
              <Skeleton className="h-3 w-1/3" />
            </div>
          ))}
        </div>
      ) : results.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-panel/40 px-6 text-center">
          <SearchX className="h-10 w-10 text-muted-foreground/60" />
          <div>
            <p className="text-sm font-semibold text-foreground">No keyframes to show</p>
            <p className="text-xs text-muted-foreground">
              {search
                ? "Try a different query or model."
                : "Run a search to see matching keyframes."}
            </p>
          </div>
        </div>
      ) : (
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${density}, minmax(0, 1fr))` }}
        >
          {results.map((result) => (
            <KeyframeCard key={result.id} result={result} onView={onView} onSimilar={onSimilar} />
          ))}
        </div>
      )}
    </div>
  );
}
