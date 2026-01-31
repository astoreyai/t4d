interface Props {
  data?: {
    recent_60s?: Array<{
      phase: string;
      items: number;
      segments_merged: number;
      t: number;
    }>;
  };
}

export function KappaSankeyView({ data }: Props) {
  const timeline = data?.recent_60s ?? [];

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-2">Consolidation Timeline</h2>
      {timeline.length === 0 ? (
        <p className="text-gray-500">No consolidation events</p>
      ) : (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {timeline.map((ev, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-sm font-mono"
            >
              <span
                className={`px-1.5 py-0.5 rounded text-xs ${
                  ev.phase === "nrem"
                    ? "bg-blue-800"
                    : ev.phase === "rem"
                    ? "bg-purple-800"
                    : "bg-red-800"
                }`}
              >
                {ev.phase.toUpperCase()}
              </span>
              <span className="text-gray-400">-{ev.t}s</span>
              <span>{ev.items} items</span>
              {ev.segments_merged > 0 && (
                <span className="text-yellow-400">
                  {ev.segments_merged} merged
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
