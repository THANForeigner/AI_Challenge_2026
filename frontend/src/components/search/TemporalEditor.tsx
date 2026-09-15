import { useState } from "react";
import { GripVertical, Plus, Trash2 } from "lucide-react";
import type { TemporalEvent } from "@/lib/pltn.types";

export function TemporalEditor({
  events,
  onChange,
}: {
  events: TemporalEvent[];
  onChange: (events: TemporalEvent[]) => void;
}) {
  const [draft, setDraft] = useState("");

  function addEvent() {
    const text = draft.trim();
    if (!text) return;
    onChange([
      ...events,
      {
        id: crypto.randomUUID(),
        text,
        order: events.length,
      },
    ]);
    setDraft("");
  }

  function removeEvent(id: string) {
    onChange(events.filter((e) => e.id !== id).map((e, i) => ({ ...e, order: i })));
  }

  function moveUp(index: number) {
    if (index === 0) return;
    const next = [...events];
    const temp = next[index]!;
    next[index] = next[index - 1]!;
    next[index - 1] = temp;
    onChange(next.map((e, i) => ({ ...e, order: i })));
  }

  return (
    <div className="space-y-2 rounded-lg border border-border bg-panel p-3">
      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Temporal events
      </p>
      <div className="flex gap-2">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") addEvent();
          }}
          placeholder="Add an event step..."
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none ring-ring focus:ring-1"
        />
        <button
          type="button"
          onClick={addEvent}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>
      <div className="space-y-1.5">
        {events.map((event, index) => (
          <div
            key={event.id}
            className="flex items-center gap-2 rounded-md border border-border bg-background px-2 py-1.5"
          >
            <GripVertical className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-secondary text-[10px] font-bold text-secondary-foreground">
              {index + 1}
            </span>
            <span className="min-w-0 flex-1 truncate text-sm text-foreground">{event.text}</span>
            <button
              type="button"
              onClick={() => moveUp(index)}
              disabled={index === 0}
              className="text-xs text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              Up
            </button>
            <button
              type="button"
              onClick={() => removeEvent(event.id)}
              className="grid h-6 w-6 shrink-0 place-items-center rounded text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
