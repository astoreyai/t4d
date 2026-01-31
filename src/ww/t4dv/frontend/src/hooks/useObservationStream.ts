import { useEffect, useRef, useState } from "react";

export interface DashboardState {
  timestamp: number;
  spiking?: any;
  storage?: any;
  consolidation?: any;
  neuromod?: any;
}

export function useObservationStream(url: string) {
  const [state, setState] = useState<DashboardState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (evt) => {
        try {
          setState(JSON.parse(evt.data));
        } catch {
          // ignore malformed frames
        }
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [url]);

  return { state, connected };
}
