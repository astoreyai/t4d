/**
 * V6.4 vis.js timeline sync
 *
 * A simplified timeline bar at the bottom of the 3D view.
 * Provides a brush range selector that filters the 3D memory space by time.
 * Full vis-timeline integration can be swapped in when the dep is installed.
 */

import { useRef, useCallback, useEffect, useState } from "react";
import type { MemoryNode } from "./MemorySpace";

interface Props {
  nodes: MemoryNode[];
  timeRange?: [number, number];
  onTimeRangeChange?: (range: [number, number]) => void;
  selectedId: string | null;
}

export function TimelineSync({
  nodes,
  timeRange,
  onTimeRangeChange,
  selectedId,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dragging, setDragging] = useState<"left" | "right" | "body" | null>(null);
  const dragStartRef = useRef(0);

  // Compute global time bounds
  const timestamps = nodes.map((n) => n.timestamp);
  const tMin = timestamps.length ? Math.min(...timestamps) : 0;
  const tMax = timestamps.length ? Math.max(...timestamps) : 1;
  const tSpan = Math.max(tMax - tMin, 1);

  const rangeStart = timeRange ? timeRange[0] : tMin;
  const rangeEnd = timeRange ? timeRange[1] : tMax;

  const toX = useCallback(
    (t: number, width: number) => ((t - tMin) / tSpan) * width,
    [tMin, tSpan],
  );
  const fromX = useCallback(
    (x: number, width: number) => tMin + (x / width) * tSpan,
    [tMin, tSpan],
  );

  // Draw timeline
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, W, H);

    // Draw event ticks
    for (const node of nodes) {
      const x = toX(node.timestamp, W);
      const isSelected = node.id === selectedId;
      ctx.fillStyle = isSelected ? "#ffffff" : "#4b5563";
      ctx.fillRect(x, 4, 1, H - 8);
    }

    // Draw selection range
    const x0 = toX(rangeStart, W);
    const x1 = toX(rangeEnd, W);
    ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
    ctx.fillRect(x0, 0, x1 - x0, H);

    // Handles
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(x0 - 2, 0, 4, H);
    ctx.fillRect(x1 - 2, 0, 4, H);
  }, [nodes, rangeStart, rangeEnd, selectedId, toX]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const W = canvas.width;
    const x0 = toX(rangeStart, W);
    const x1 = toX(rangeEnd, W);

    if (Math.abs(x - x0) < 8) {
      setDragging("left");
    } else if (Math.abs(x - x1) < 8) {
      setDragging("right");
    } else if (x > x0 && x < x1) {
      setDragging("body");
      dragStartRef.current = x;
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragging || !canvasRef.current || !onTimeRangeChange) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const W = canvasRef.current.width;
    const t = fromX(x, W);

    if (dragging === "left") {
      onTimeRangeChange([Math.min(t, rangeEnd - 0.001), rangeEnd]);
    } else if (dragging === "right") {
      onTimeRangeChange([rangeStart, Math.max(t, rangeStart + 0.001)]);
    } else if (dragging === "body") {
      const dx = x - dragStartRef.current;
      const dt = (dx / W) * tSpan;
      dragStartRef.current = x;
      onTimeRangeChange([rangeStart + dt, rangeEnd + dt]);
    }
  };

  const handleMouseUp = () => setDragging(null);

  return (
    <div className="bg-gray-900 rounded-b-lg px-2 py-1">
      <div className="flex items-center gap-2 text-xs text-gray-400 mb-1">
        <span>Timeline</span>
        <span className="ml-auto">
          {new Date(rangeStart).toLocaleTimeString()} –{" "}
          {new Date(rangeEnd).toLocaleTimeString()}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={800}
        height={32}
        className="w-full rounded cursor-col-resize"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  );
}
