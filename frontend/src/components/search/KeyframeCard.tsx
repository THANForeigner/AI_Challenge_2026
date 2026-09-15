import { Eye, ImageIcon } from "lucide-react";
import type { KeyframeResult } from "@/lib/pltn.types";

interface KeyframeCardProps {
  result: KeyframeResult;
  onView: (result: KeyframeResult) => void;
  onSimilar: (result: KeyframeResult) => void;
}

export function KeyframeCard({ result, onView, onSimilar }: KeyframeCardProps) {
  return (
    <div className="group relative overflow-hidden rounded-lg border border-border bg-card transition-all hover:border-primary hover:shadow-[0_0_16px_oklch(0.72_0.22_345/25%)]">
      <div className="relative aspect-video w-full overflow-hidden bg-black">
        <img
          src={result.imageUrl}
          alt={`${result.videoId} frame ${result.frameIdx}`}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        <div className="absolute inset-0 flex items-center justify-center gap-2 bg-black/60 opacity-0 transition-opacity group-hover:opacity-100">
          <button
            type="button"
            onClick={() => onView(result)}
            className="grid h-9 w-9 place-items-center rounded-full bg-primary text-primary-foreground hover:bg-primary/90"
            title="View details"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onSimilar(result)}
            className="grid h-9 w-9 place-items-center rounded-full bg-secondary text-secondary-foreground hover:bg-secondary/80"
            title="Find similar"
          >
            <ImageIcon className="h-4 w-4" />
          </button>
        </div>
        <div className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-mono text-neon-light-pink">
          #{result.rank}
        </div>
        {result.source === "demo" && (
          <div className="absolute right-2 top-2 rounded bg-neon-amber/90 px-1.5 py-0.5 text-[10px] font-bold text-black">
            DEMO
          </div>
        )}
      </div>
      <div className="space-y-1 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-xs font-semibold text-foreground">
            {result.videoId}
          </span>
          <span className="shrink-0 text-[10px] text-muted-foreground">f{result.frameIdx}</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-neon-pink">{(result.score * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
