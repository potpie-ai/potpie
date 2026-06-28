import type {
  GraphData,
  PotsResponse,
  SearchEntity,
  StatusResponse,
} from "./types";

const BASE = "/ui/api";

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body?.detail || `request failed (${res.status})`);
  }
  return body as T;
}

function potParam(pot?: string): string {
  return pot ? `pot=${encodeURIComponent(pot)}` : "";
}

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
      throw new Error(b?.detail || `switch failed (${res.status})`);
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

  catalog: (pot?: string) =>
    jget<Record<string, unknown>>(`/catalog${pot ? `?${potParam(pot)}` : ""}`),
};
