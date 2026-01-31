import { useObservationStream } from "./hooks/useObservationStream";
import { SpikeRasterView } from "./components/SpikeRasterView";
import { KappaSankeyView } from "./components/KappaSankeyView";
import { T4dxDashboard } from "./components/T4dxDashboard";

export default function App() {
  const { state, connected } = useObservationStream("ws://localhost:8420/ws/observe");

  return (
    <div className="min-h-screen p-4">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">T4DV Dashboard</h1>
        <span
          className={`px-2 py-1 rounded text-sm ${
            connected ? "bg-green-800" : "bg-red-800"
          }`}
        >
          {connected ? "Connected" : "Disconnected"}
        </span>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SpikeRasterView data={state?.spiking} />
        <KappaSankeyView data={state?.consolidation} />
        <T4dxDashboard data={state?.storage} />
      </div>
    </div>
  );
}
