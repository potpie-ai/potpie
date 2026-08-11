import type {
  GraphData,
  Origin,
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

/** Build a query string from defined params only.
 *
 * Every read is scoped by `host` as well as `pot`: a pot id means nothing
 * without the host it was listed from, so the two always travel together.
 */
function qs(params: Record<string, string | number | undefined>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

export const api = {
  pots: () => jget<PotsResponse>("/pots"),

  usePot: async (ref: string, host?: Origin) => {
    const res = await fetch(`${BASE}/pots/use`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref, host }),
    });
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      throw new Error(errorDetail(b?.detail, `switch failed (${res.status})`));
    }
    return res.json();
  },

  status: (pot?: string, host?: Origin) =>
    jget<StatusResponse>(`/status${qs({ pot, host })}`),

  graph: (pot?: string, host?: Origin) =>
    jget<GraphData>(`/graph${qs({ pot, host })}`),

  neighborhood: (key: string, depth: number, pot?: string, host?: Origin) =>
    jget<GraphData>(`/neighborhood${qs({ key, depth, pot, host })}`),

  search: (q: string, pot?: string, host?: Origin) =>
    jget<{ entities: SearchEntity[] }>(
      `/search${qs({ q, limit: 20, pot, host })}`,
    ),
};
