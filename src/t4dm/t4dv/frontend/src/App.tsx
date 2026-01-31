import { useState } from "react";
import { useObservationStream } from "./hooks/useObservationStream";
import { SpikeRasterView } from "./components/SpikeRasterView";
import { KappaSankeyView } from "./components/KappaSankeyView";
import { T4dxDashboard } from "./components/T4dxDashboard";
import { MemorySpace, type ProjectionMode, type MemoryNode, type MemoryEdge } from "./components/three";

// Demo data — in production, fetched from /api/v1/viz/snapshot/storage
const DEMO_NODES: MemoryNode[] = [];
const DEMO_EDGES: MemoryEdge[] = [];

export default function App() {
  const { state, connected } = useObservationStream("ws://localhost:8420/ws/observe");
  const [tab, setTab] = useState<"2d" | "3d">("2d");
  const [projMode, setProjMode] = useState<ProjectionMode>("slice");
  const [timeRange, setTimeRange] = useState<[number, number] | undefined>();

  return (
    <div className="min-h-screen p-4">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">T4DV Dashboard</h1>
        <div className="flex items-center gap-3">
          <div className="flex rounded overflow-hidden text-sm">
            <button
              onClick={() => setTab("2d")}
              className={`px-3 py-1 ${tab === "2d" ? "bg-blue-700" : "bg-gray-700"}`}
            >
              2D
            </button>
            <button
              onClick={() => setTab("3d")}
              className={`px-3 py-1 ${tab === "3d" ? "bg-blue-700" : "bg-gray-700"}`}
            >
              3D
            </button>
          </div>
          {tab === "3d" && (
            <select
              value={projMode}
              onChange={(e) => setProjMode(e.target.value as ProjectionMode)}
              className="bg-gray-700 rounded text-sm px-2 py-1"
            >
              <option value="slice">Slice</option>
              <option value="collapse">Collapse</option>
              <option value="animate">Animate</option>
            </select>
          )}
          <span
            className={`px-2 py-1 rounded text-sm ${
              connected ? "bg-green-800" : "bg-red-800"
            }`}
          >
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </header>

      {tab === "2d" ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <SpikeRasterView data={state?.spiking} />
          <KappaSankeyView data={state?.consolidation} />
          <T4dxDashboard data={state?.storage} />
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg" style={{ height: "70vh" }}>
          <MemorySpace
            nodes={DEMO_NODES}
            edges={DEMO_EDGES}
            projectionMode={projMode}
            timeRange={timeRange}
            onTimeRangeChange={setTimeRange}
          />
        </div>
      )}
    </div>
  );
}
