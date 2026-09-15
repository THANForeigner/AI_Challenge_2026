export function Logo({ className }: { className?: string }) {
  return (
    <div
      className={`flex items-center justify-center rounded-lg border border-neon-pink/30 bg-black/40 px-4 py-3 ${className ?? ""}`}
      style={{ boxShadow: "var(--glow-pink)" }}
    >
      <span
        className="text-2xl font-black tracking-[0.12em] text-neon-pink"
        style={{
          textShadow: "0 0 12px oklch(0.72 0.24 345 / 80%)",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        }}
      >
        TongFC
      </span>
    </div>
  );
}
