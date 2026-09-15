import type { SearchMode } from "@/lib/pltn.types";

const MODES: { value: SearchMode; label: string }[] = [
  { value: "text", label: "Text" },
  { value: "image", label: "Image" },
  { value: "temporal", label: "Temporal" },
];

export function ModeTabs({
  value,
  onChange,
}: {
  value: SearchMode;
  onChange: (mode: SearchMode) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-1.5 rounded-lg border border-border bg-panel p-1.5">
      {MODES.map((mode) => {
        const active = mode.value === value;
        return (
          <button
            key={mode.value}
            type="button"
            onClick={() => onChange(mode.value)}
            className={`rounded-md px-2 py-1.5 text-xs font-semibold transition-all ${
              active
                ? "bg-primary text-primary-foreground shadow-[0_0_12px_oklch(0.72_0.22_345/40%)]"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            }`}
          >
            {mode.label}
          </button>
        );
      })}
    </div>
  );
}
