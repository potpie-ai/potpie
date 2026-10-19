import type {
  GraphData,
  Origin,
  PotsResponse,
  SearchEntity,
  StatusResponse,
} from "./types";
import type {
  CommitDetail,
  CommitPage,
  JournalStatus,
  PlanHistoryPage,
  PreviewResult,
} from "./commitTypes";
import { session, SessionRequiredError } from "./session.ts";

const BASE = "/ui/api";

function errorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail === "object" && "reasons" in detail) {
    const reasons = (detail as { reasons: { message: string }[] }).reasons;
    if (Array.isArray(reasons) && reasons.length) {
      return reasons.map((reason) => reason.message).join("; ");
    }
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

/**
 * Every call to the daemon goes through here.
 *
 * `credentials: "same-origin"` sends the HttpOnly session cookie `potpie ui`
 * handed this browser (and nothing to any other origin). A 401 means that
 * session is gone; it is reported once to the session store, which switches
 * the page to its "run `potpie ui` again" state.
 */
async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: "same-origin",
  });
  if (res.status === 401) {
    session.markRequired();
    throw new SessionRequiredError();
  }
  return res;
}

/** A failed call, carrying the daemon's machine-readable refusal code
 * (`detail.status`, e.g. `preview_stale`) so callers can react to it. */
export class CommitApiError extends Error {
  // A declared field, not a constructor parameter property: the tests run this
  // module through Node's type stripping, which cannot rewrite those.
  readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "CommitApiError";
    this.code = code;
  }
}

async function failedResponse(res: Response): Promise<never> {
  const body = await res.json().catch(() => ({}));
  throw new CommitApiError(
    errorDetail(body?.detail, `request failed (${res.status})`),
    body?.detail?.status || "request_failed",
  );
}

async function jget<T>(path: string): Promise<T> {
  const res = await request(path);
  if (!res.ok) return failedResponse(res);
  return res.json() as Promise<T>;
}

async function jpost<T>(path: string, payload: object): Promise<T> {
  const res = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) return failedResponse(res);
  return res.json() as Promise<T>;
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

let sessionReported = false;

export const api = {
  mutationHistory: (pot: string, host?: Origin, limit = 50) =>
    jget<PlanHistoryPage>(`/mutation-history${qs({ pot, host, limit })}`),

  commits: (pot: string, host?: Origin, cursor?: string) =>
    jget<CommitPage>(`/commits${qs({ pot, host, cursor })}`),

  commit: (commit_id: string, pot: string, host?: Origin, offset = 0) =>
    jget<CommitDetail>(`/commit${qs({ commit_id, pot, host, offset })}`),

  journal: (pot: string, host?: Origin) =>
    jget<JournalStatus>(`/journal${qs({ pot, host })}`),

  /** A dry run: the daemon computes what the revert or rollback would change
   * and returns a short-lived preview id. Nothing in the graph changes. */
  preview: (
    target_commit_id: string,
    mode: "revert" | "rollback",
    expected_head: string,
    pot: string,
    host?: Origin,
  ) =>
    jpost<PreviewResult>("/rollback/preview", {
      target_commit_id,
      expected_head,
      mode,
      pot,
      host,
    }),

  pots: () => jget<PotsResponse>("/pots"),

  usePot: async (ref: string, host?: Origin) => {
    const res = await request("/pots/use", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(host ? { ref, host } : { ref }),
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

  reportSession: (hadGraph: boolean) => {
    if (sessionReported) return;
    sessionReported = true;
    void request("/telemetry/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ had_graph: hadGraph }),
    }).catch(() => undefined);
  },
};
