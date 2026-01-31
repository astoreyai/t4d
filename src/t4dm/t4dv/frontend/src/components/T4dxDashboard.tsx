import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface Props {
  data?: {
    windows?: Record<
      string,
      { ops_total: number; ops_by_type?: Record<string, number>; total_duration_ms?: number }
    >;
    segment_count?: number;
    memtable_count?: number;
  };
}

export function T4dxDashboard({ data }: Props) {
  const win10s = data?.windows?.["10s"];
  const opsByType = win10s?.ops_by_type ?? {};

  const chartData = Object.entries(opsByType).map(([op, count]) => ({
    op,
    count,
  }));

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-2">T4DX Storage</h2>
      <div className="flex gap-4 text-sm text-gray-400 mb-3">
        <span>Segments: {data?.segment_count ?? 0}</span>
        <span>MemTable: {data?.memtable_count ?? 0}</span>
        <span>Ops (10s): {win10s?.ops_total ?? 0}</span>
      </div>
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={chartData} layout="vertical">
            <XAxis type="number" stroke="#6b7280" />
            <YAxis dataKey="op" type="category" stroke="#6b7280" width={100} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <p className="text-gray-500">No storage operations</p>
      )}
    </div>
  );
}
