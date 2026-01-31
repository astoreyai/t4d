import { useEffect, useRef } from "react";

interface Props {
  data?: {
    blocks?: Record<number, { firing_rate: number; pe: number; goodness: number }>;
    windows?: Record<string, { count: number; mean_firing_rate?: number }>;
  };
}

export function SpikeRasterView({ data }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<Array<Record<number, number>>>([]);

  useEffect(() => {
    if (!data?.blocks || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Append current block firing rates to history
    const frame: Record<number, number> = {};
    for (const [idx, block] of Object.entries(data.blocks)) {
      frame[Number(idx)] = block.firing_rate;
    }
    historyRef.current.push(frame);
    if (historyRef.current.length > canvas.width) {
      historyRef.current = historyRef.current.slice(-canvas.width);
    }

    // Render
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const numBlocks = 6;
    const rowH = canvas.height / numBlocks;
    const history = historyRef.current;

    for (let t = 0; t < history.length; t++) {
      for (let b = 0; b < numBlocks; b++) {
        const rate = history[t][b] ?? 0;
        const intensity = Math.min(255, Math.floor(rate * 255));
        ctx.fillStyle = `rgb(${intensity}, ${intensity >> 1}, 0)`;
        ctx.fillRect(t, b * rowH, 1, rowH);
      }
    }
  }, [data]);

  const win1s = data?.windows?.["1s"];

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-2">Spike Raster</h2>
      <canvas
        ref={canvasRef}
        width={600}
        height={180}
        className="w-full rounded"
      />
      {win1s && (
        <p className="text-sm text-gray-400 mt-1">
          1s: {win1s.count} spikes, mean rate:{" "}
          {(win1s.mean_firing_rate ?? 0).toFixed(3)}
        </p>
      )}
    </div>
  );
}
