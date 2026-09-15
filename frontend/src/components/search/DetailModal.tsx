import { ChevronLeft, ChevronRight, Copy, ImageIcon, Timer } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { formatTimestamp } from "@/lib/pltn-format";
import type { KeyframeResult } from "@/lib/pltn.types";

interface DetailModalProps {
  result: KeyframeResult | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onPrev: () => void;
  onNext: () => void;
  canPrev: boolean;
  canNext: boolean;
  onCopySubmit: (result: KeyframeResult) => void;
  onSimilar: (result: KeyframeResult) => void;
}

export function DetailModal({
  result,
  open,
  onOpenChange,
  onPrev,
  onNext,
  canPrev,
  canNext,
  onCopySubmit,
  onSimilar,
}: DetailModalProps) {
  if (!result) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl gap-4 p-0 sm:rounded-xl">
        <DialogHeader className="px-5 pt-5">
          <DialogTitle className="font-mono text-base">
            {result.videoId}{" "}
            <span className="text-muted-foreground">· frame {result.frameIdx}</span>
          </DialogTitle>
        </DialogHeader>

        <div className="relative px-5">
          <div className="relative overflow-hidden rounded-lg border border-border bg-black">
            <img
              src={result.imageUrl}
              alt={`${result.videoId} frame ${result.frameIdx}`}
              className="aspect-video w-full object-contain"
            />
            <button
              type="button"
              onClick={onPrev}
              disabled={!canPrev}
              className="absolute left-2 top-1/2 grid h-9 w-9 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-foreground backdrop-blur-sm transition-colors hover:bg-primary hover:text-primary-foreground disabled:opacity-30"
              title="Previous result"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <button
              type="button"
              onClick={onNext}
              disabled={!canNext}
              className="absolute right-2 top-1/2 grid h-9 w-9 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-foreground backdrop-blur-sm transition-colors hover:bg-primary hover:text-primary-foreground disabled:opacity-30"
              title="Next result"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
            <div className="absolute left-3 top-3 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-mono text-neon-light-pink">
              #{result.rank}
            </div>
            <div className="absolute right-3 top-3 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-mono text-neon-light-pink">
              <Timer className="h-3 w-3" />
              {formatTimestamp(result.timestamp ?? result.frameIdx / 25)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1 px-5 text-xs sm:grid-cols-4">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Video ID</p>
            <p className="font-mono text-foreground">{result.videoId}</p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Frame</p>
            <p className="font-mono text-foreground">{result.frameIdx}</p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Score</p>
            <p className="font-mono text-neon-pink">{(result.score * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Timestamp</p>
            <p className="font-mono text-foreground">
              {formatTimestamp(result.timestamp ?? result.frameIdx / 25)}
            </p>
          </div>
        </div>

        {(result.ocr || result.asr) && (
          <div className="space-y-2 px-5">
            {result.ocr && (
              <div className="rounded-md border border-border bg-panel p-2.5 text-xs">
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-neon-light-pink">
                  OCR
                </p>
                <p className="text-muted-foreground">{result.ocr}</p>
              </div>
            )}
            {result.asr && (
              <div className="rounded-md border border-border bg-panel p-2.5 text-xs">
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-neon-pink">
                  ASR
                </p>
                <p className="text-muted-foreground">{result.asr}</p>
              </div>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2 border-t border-border px-5 py-4">
          <button
            type="button"
            onClick={() => onCopySubmit(result)}
            className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <Copy className="h-3.5 w-3.5" />
            Copy video_id, frame
          </button>
          <button
            type="button"
            onClick={() => onSimilar(result)}
            className="flex items-center gap-1.5 rounded-md bg-secondary px-3 py-2 text-xs font-semibold text-secondary-foreground transition-colors hover:bg-secondary/80"
          >
            <ImageIcon className="h-3.5 w-3.5" />
            Find similar
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
