import type {
  GraphData,
  Origin,
  PotsResponse,
  SearchEntity,
  StatusResponse,
} from "./types";

import type { CommitDetail, CommitHeader, CommitPage, JournalStatus, PreviewResult } from "./commitTypes";
import type { PlanHistoryPage } from "./commitTypes";

const BASE = "/ui/api";

function errorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail === "object" && "reasons" in detail) {
    const reasons = (detail as { reasons: { message: string }[] }).reasons;
    return reasons.map(reason => reason.message).join("; ");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    const message = (detail as { message: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  if (detail !== undefined && detail !== null) {
    try {
      return JSON.stringify(detail);
    } catch {
      // Fall through to the request-specific message.
    }
  }
  return fallback;
}

export class CommitApiError extends Error {
  constructor(message: string, readonly code: string) { super(message); }
}

async function failedResponse(response: Response): Promise<never> {
  const body = await response.json().catch(() => ({}));
  throw new CommitApiError(errorDetail(body?.detail, `request failed (${response.status})`), body?.detail?.status || "request_failed");
}

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) return failedResponse(res);
  return res.json() as Promise<T>;
}

async function jpost<T>(path: string, payload: object): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) return failedResponse(response);
  return response.json() as Promise<T>;
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
  mutationHistory: (pot: string, host: Origin, limit = 50) => jget<PlanHistoryPage>(`/mutation-history${qs({ pot, host, limit })}`),
  commits: (pot: string, host: Origin, cursor?: string) => jget<CommitPage>(`/commits${qs({ pot, host, cursor })}`),
  commit: (commit_id: string, pot: string, host: Origin, offset = 0) => jget<CommitDetail>(`/commit${qs({ pot, host, commit_id, offset })}`),
  journal: (pot: string, host: Origin) => jget<JournalStatus>(`/journal${qs({ pot, host })}`),
  preview: (target_commit_id: string, mode: "revert" | "rollback", expected_head: string, pot: string, host: Origin) => jpost<PreviewResult>("/rollback/preview", { target_commit_id, mode, expected_head, pot, host }),
  applyPreview: (preview_id: string, pot: string, host: Origin) => jpost<{ commit: CommitHeader }>("/rollback/apply", { preview_id, pot, host }),
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
