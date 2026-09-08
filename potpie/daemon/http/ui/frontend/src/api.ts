import type {
  GraphData,
  PotsResponse,
  SearchEntity,
  StatusResponse,
} from "./types";

const BASE = "/ui/api";

function errorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (detail !== undefined && detail !== null) {
    try {
      return JSON.stringify(detail);
    } catch {
      // Fall through to the request-specific message.
    }
  }
  return fallback;
}

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(errorDetail(body?.detail, `request failed (${res.status})`));
  }
  return body as T;
}

function potParam(pot?: string): string {
  return pot ? `pot=${encodeURIComponent(pot)}` : "";
}

let sessionReported = false;

export const api = {
  pots: () => jget<PotsResponse>("/pots"),

  usePot: async (ref: string) => {
    const res = await fetch(`${BASE}/pots/use`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref }),
    });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      throw new Error(errorDetail(b?.detail, `switch failed (${res.status})`));
    }
    return res.json();
  },

  status: (pot?: string) =>
    jget<StatusResponse>(`/status${pot ? `?${potParam(pot)}` : ""}`),

  graph: (pot?: string) =>
    jget<GraphData>(`/graph${pot ? `?${potParam(pot)}` : ""}`),

  neighborhood: (key: string, depth: number, pot?: string) =>
    jget<GraphData>(
      `/neighborhood?key=${encodeURIComponent(key)}&depth=${depth}${
        pot ? `&${potParam(pot)}` : ""
      }`,
    ),

  search: (q: string, pot?: string) =>
    jget<{ entities: SearchEntity[] }>(
      `/search?q=${encodeURIComponent(q)}&limit=20${
        pot ? `&${potParam(pot)}` : ""
      }`,
    ),

  reportSession: (hadGraph: boolean) => {
    if (sessionReported) return;
    sessionReported = true;
    void fetch(`${BASE}/telemetry/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ had_graph: hadGraph }),
    }).catch(() => undefined);
  },
};
