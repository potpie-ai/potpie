import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { CommitIcon } from "./CommitPanels";
import type { Origin } from "./types";

import type { PlanHistoryPage, SavedPlan } from "./commitTypes";

const planTitle = (plan: SavedPlan) => {
  const diff = plan.payload.diff;
  return plan.status === "committed"
    ? `${diff.claims_asserted} claims added · ${diff.claims_retracted} retracted`
    : `${diff.claims_asserted} claims to add · ${diff.claims_retracted} to retract`;
};

export default function PlanHistory({ pot, host }: { pot: string; host: Origin }) {
  const [page, setPage] = useState<PlanHistoryPage | null>(null);
  const [selected, setSelected] = useState<string>();
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(50);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const request = useRef(0);
  const load = useCallback(async (count: number) => {
    const token = ++request.current;
    setBusy(true);
    setError("");
    try {
      const next = await api.mutationHistory(pot, host, count);
      if (request.current !== token) return;
      setPage(next);
      setLimit(count);
      setSelected(previous => next.entries.some(plan => plan.id === previous)
        ? previous : (next.entries.find(plan => plan.status === "committed") ?? next.entries[0])?.id);
    } catch (cause) {
      if (request.current === token)
        setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request.current === token) setBusy(false);
    }
  }, [pot, host]);

  useEffect(() => {
    void load(50);
    return () => { request.current++; };
  }, [load]);

  const rows = page?.entries.filter(plan =>
    `${plan.id} ${plan.mutation_id ?? ""} ${plan.status} ${plan.entity_keys.join(" ")} ${plan.source_refs.join(" ")}`
      .toLowerCase().includes(query.trim().toLowerCase())) ?? [];
  const detail = page?.entries.find(plan => plan.id === selected);

  return (
    <section className="commits-view" aria-label="Saved mutation plans">
      <aside className="commit-history" aria-label="Saved plan history">
        <div className="commit-history-head">
          <div className="commit-section-heading">
            <h1>Mutation history</h1>
            <button className="commit-button icon-only" aria-label="Refresh history"
              disabled={busy} onClick={() => void load(limit)}>
              <CommitIcon name="refresh" />
            </button>
          </div>
          <p>Saved mutation plans</p>
          <label className="commit-filter">
            <CommitIcon name="search" />
            <input aria-label="Filter saved plans" placeholder="Filter saved plans…"
              value={query} onChange={event => setQuery(event.target.value)} />
          </label>
        </div>
        <div className="commit-history-scroll" aria-busy={busy}>
          {busy && <p className="commit-empty-note" role="status">Loading saved plans…</p>}
          {!busy && rows.length === 0 && <p className="commit-empty-note">
            {query ? "No loaded plans match your search." : error ? "History could not be loaded. Try refreshing." : "No saved mutation plans yet."}
          </p>}
          {rows.map(plan => (
            <button key={plan.id} className={`commit-row${selected === plan.id ? " on" : ""}`}
              aria-current={selected === plan.id ? "true" : undefined}
              onClick={() => setSelected(plan.id)}>
              <span className="commit-rail"><CommitIcon name="commit" /></span>
              <span className="commit-row-body">
                <span className="commit-row-top">
                  <span>{new Date(plan.occurred_at).toLocaleString()}</span>
                  <span className="commit-tag">{plan.status}</span>
                </span>
                <strong>{planTitle(plan)}</strong>
                <span className="commit-row-meta">{plan.id}</span>
              </span>
            </button>
          ))}
          {page && page.entries.length === limit && limit < 200 && (
            <button className="commit-button commit-load-more" disabled={busy}
              onClick={() => void load(Math.min(limit + 50, 200))}>Load more plans</button>
          )}
        </div>
        <div className="commit-history-foot">Showing up to {limit} recent saved plans</div>
      </aside>
      <div className="commit-detail">
        <div className="commit-detail-content saved-plan-detail">
          <p className="commit-notice warning" role="status">
            Native journal coverage has not started for this pot. Saved plans show
            mutation status and scope. Before/after comparisons and rollback are unavailable.
          </p>
          {error && <p className="commit-notice error-notice" role="alert">{error}</p>}
          {page?.warnings.map(warning => <p key={warning} className="commit-notice warning">{warning}</p>)}
          {detail && <>
            <div>
              <span className="commit-tag">Saved plan · {detail.status}</span>
              <h2>{planTitle(detail)}</h2>
              <p className="saved-plan-id">{detail.id}</p>
              <p>{new Date(detail.occurred_at).toLocaleString()}</p>
              {detail.mutation_id && <p className="saved-plan-id">Mutation: {detail.mutation_id}</p>}
              {detail.detail && <p>{detail.detail}</p>}
            </div>
            <div>
              <h3>Planned changes</h3>
              <p>{detail.payload.diff.entity_upserts} entity upserts · {detail.payload.diff.edge_upserts} relationship upserts · {detail.payload.diff.invalidations} invalidations</p>
              <p>{detail.payload.accepted_operations.length} accepted operations across {[...new Set(detail.payload.accepted_operations.map(op => op.subgraph))].join(", ") || "no subgraphs"}</p>
            </div>
            <details open>
              <summary>Entities in this plan ({detail.entity_keys.length})</summary>
              <ul>{detail.entity_keys.map(key => <li key={key}>{key}</li>)}</ul>
            </details>
            <details>
              <summary>Source references ({detail.source_refs.length})</summary>
              <ul>{detail.source_refs.map(ref => <li key={ref}>{ref}</li>)}</ul>
            </details>
          </>}
        </div>
      </div>
    </section>
  );
}
