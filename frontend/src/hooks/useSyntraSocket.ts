
import { useEffect, useState } from 'react';
import type { Telemetry } from '../types/syntra';

export function useSyntraSocket() {
  const [data, setData] = useState<Telemetry | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number;
    let alive = true;

    const connect = () => {
      const scheme =
        location.protocol === 'https:' ? 'wss' : 'ws';

      ws = new WebSocket(
        `${scheme}://${location.host}/ws/telemetry`
      );

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const telemetry: Telemetry = JSON.parse(event.data);
          setData(telemetry);
        } catch {
          // Ignore invalid telemetry messages.
        }
      };

      ws.onclose = () => {
        setConnected(false);

        if (alive) {
          timer = window.setTimeout(connect, 2500);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      alive = false;
      clearTimeout(timer);
      ws?.close();
    };
  }, []);

  return { data, connected };
}
