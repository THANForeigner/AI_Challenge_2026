import { useState } from "react";
import { Settings } from "lucide-react";

export function ApiSettings({
  baseUrl,
  onChange,
}: {
  baseUrl: string;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-border bg-panel p-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wider text-muted-foreground"
      >
        <span className="flex items-center gap-1.5">
          <Settings className="h-3.5 w-3.5" />
          API Settings
        </span>
        <span className="text-[10px]">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <p className="text-[10px] leading-relaxed text-muted-foreground">
            Set your search API base URL. Leave empty to use demo data.
          </p>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => onChange(e.target.value)}
            placeholder="http://localhost:8000"
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-xs text-foreground outline-none ring-ring focus:ring-1"
          />
          <p className="text-[10px] text-muted-foreground">
            Also configurable via{" "}
            <code className="rounded bg-background px-1 py-0.5 text-neon-pink">
              PLTN_API_BASE_URL
            </code>{" "}
            env var.
          </p>
        </div>
      )}
    </div>
  );
}
