import { useCallback, useEffect, useRef, useState } from "react";
import { api, CommitApiError } from "./api";
import {
  CommitHistory,
  CommitIcon,
  CommitInspector,
  RollbackPreview,
} from "./CommitPanels";
import "./commits.css";
import type {
  CommitDetail,
  CommitHeader,
  CommitPage,
  JournalStatus,
  PreviewResult,
} from "./commitTypes";
import type { GraphData, Origin } from "./types";
import { mergeContext } from "./mutationContext";
import PlanHistory from "./PlanHistory";

interface Props {
  pot: string;
  host: Origin;
  onApplied: () => void;
}
const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export default function CommitView({ pot, host, onApplied }: Props) {
  const [page, setPage] = useState<CommitPage | null>(null);
  const [journal, setJournal] = useState<JournalStatus | null>(null);
  const [detail, setDetail] = useState<CommitDetail | null>(null);
  const [context, setContext] = useState<GraphData>({ nodes: [], edges: [] });
  const [contextError, setContextError] = useState("");
  const [previewMode, setPreviewMode] = useState<"revert" | "rollback">(
    "revert",
  );
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [outdated, setOutdated] = useState(false);
  const [selectedCommit, setSelectedCommit] = useState<string | undefined>();
  const [selectedRecord, setSelectedRecord] = useState<string | null>(null);
  const alive = useRef(true);
  const request = useRef(0);
  const historyRequest = useRef(0);
  const loading = useRef(false);

  const reload = useCallback(async () => {
    const token = ++historyRequest.current;
    let loaded;
    try {
      loaded = await Promise.all([
        api.commits(pot, host),
        api.journal(pot, host),
        api
          .graph(pot, host)
          .then((data) => ({ data, error: "" }))
          .catch((error) => ({
            data: { nodes: [], edges: [] },
            error: errorText(error),
          })),
      ]);
    } catch (e) {
      // A host older than commit history still has saved plans to show.
      if (!(e instanceof CommitApiError) || e.code !== "host_outdated") throw e;
      if (alive.current && historyRequest.current === token) setOutdated(true);
      return null;
    }
    const [commits, status, graph] = loaded;
    if (alive.current && historyRequest.current === token) {
      setOutdated(false);
      setPage(commits);
      setJournal(status);
      setContext(graph.data);
      setContextError(graph.error);
      return commits;
    }
    return null;
  }, [pot, host]);
  const inspect = useCallback(
    async (row: CommitHeader) => {
      const current = ++request.current;
      setSelectedCommit(row.commit_id);
      setBusy(true);
      setError("");
      setNotice("");
      setPreview(null);
      setSelectedRecord(null);
      setDetail(null);
      try {
        let next = await api.commit(row.commit_id, pot, host);
        // Receipts paginate effects separately from records. Load the complete
        // normal-sized receipt so audit pages cannot change the visual classification.
        for (let count = 1; next.next_offset !== null && count < 20; count++) {
          if (!alive.current || request.current !== current) return;
          const offset = next.next_offset;
          const part = await api.commit(row.commit_id, pot, host, offset);
          if (part.next_offset !== null && part.next_offset <= offset)
            throw new Error(
              "The commit returned an invalid continuation offset.",
            );
          next = {
            ...part,
            changes: [...next.changes, ...part.changes],
            display_context: [...next.display_context, ...part.display_context],
          };
        }
        if (alive.current && request.current === current) setDetail(next);
      } catch (e) {
        if (alive.current && request.current === current)
          setError(errorText(e));
      } finally {
        if (alive.current && request.current === current) setBusy(false);
      }
    },
    [pot, host],
  );

  useEffect(() => {
    alive.current = true;
    reload()
      .then((commits) => {
        if (!commits) return;
        if (commits.headers[0]) return inspect(commits.headers[0]);
        setBusy(false);
      })
      .catch((e) => {
        if (alive.current) {
          setError(errorText(e));
          setBusy(false);
        }
      });
    return () => {
      alive.current = false;
      request.current++;
      historyRequest.current++;
    };
  }, [reload, inspect]);

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      await reload();
    } catch (e) {
      if (alive.current) setError(errorText(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function loadMore() {
    if (!page?.next_cursor || loading.current) return;
    loading.current = true;
    const token = ++historyRequest.current;
    const cursor = page.next_cursor;
    setBusy(true);
    setError("");
    try {
      const next = await api.commits(pot, host, cursor);
      if (alive.current && historyRequest.current === token)
        setPage((previous) =>
          previous?.next_cursor === cursor
            ? {
                ...next,
                headers: [
                  ...new Map(
                    [...previous.headers, ...next.headers].map((row) => [
                      row.commit_id,
                      row,
                    ]),
                  ).values(),
                ],
              }
            : previous,
        );
    } catch (e) {
      if (alive.current) setError(errorText(e));
    } finally {
      loading.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function moreDetail() {
    if (!detail || detail.next_offset === null || loading.current) return;
    loading.current = true;
    const token = ++request.current;
    const commitId = detail.header.commit_id;
    setBusy(true);
    setError("");
    try {
      const next = await api.commit(
        detail.header.commit_id,
        pot,
        host,
        detail.next_offset,
      );
      if (alive.current && request.current === token)
        setDetail((previous) =>
          previous?.header.commit_id === commitId
            ? {
                ...next,
                changes: [...previous.changes, ...next.changes],
                display_context: [
                  ...previous.display_context,
                  ...next.display_context,
                ],
              }
            : previous,
        );
    } catch (e) {
      if (alive.current) setError(errorText(e));
    } finally {
      loading.current = false;
      if (alive.current && request.current === token) setBusy(false);
    }
  }
  async function generatePreview(
    mode: "revert" | "rollback",
    target: string,
    head: string,
    token: number,
  ) {
    try {
      const next = await api.preview(target, mode, head, pot, host);
      if (alive.current && request.current === token) setPreview(next);
    } catch (e) {
      if (!(e instanceof CommitApiError) || e.code !== "preview_stale") throw e;
      const fresh = await reload();
      if (!alive.current || request.current !== token || !fresh?.coverage.head)
        return;
      const next = await api.preview(
        target,
        mode,
        fresh.coverage.head,
        pot,
        host,
      );
      if (alive.current && request.current === token) {
        setPreview(next);
        setNotice(
          "Graph changed. Review the regenerated preview before applying.",
        );
      }
    }
  }
  async function makePreview(mode: "revert" | "rollback") {
    if (!detail || !page?.coverage.head || loading.current) return;
    loading.current = true;
    const token = ++request.current;
    setBusy(true);
    setError("");
    setPreview(null);
    setNotice("");
    setPreviewMode(mode);
    try {
      await generatePreview(
        mode,
        detail.header.commit_id,
        page.coverage.head,
        token,
      );
    } catch (e) {
      if (alive.current && request.current === token) setError(errorText(e));
    } finally {
      loading.current = false;
      if (alive.current && request.current === token) setBusy(false);
    }
  }
  async function apply() {
    if (!preview || !detail || loading.current) return;
    loading.current = true;
    const token = ++request.current;
    setBusy(true);
    setError("");
    try {
      const result = await api.applyPreview(
        preview.preview.preview_id,
        pot,
        host,
      );
      if (!alive.current || request.current !== token) return;
      setPreview(null);
      setNotice(
        `Committed #${result.commit.sequence}. Select the new commit to undo it.`,
      );
      onApplied();
      await reload();
    } catch (e) {
      if (!alive.current || request.current !== token) return;
      if (e instanceof CommitApiError && e.code === "preview_stale") {
        setPreview(null);
        try {
          const fresh = await reload();
          if (fresh?.coverage.head)
            await generatePreview(
              previewMode,
              detail.header.commit_id,
              fresh.coverage.head,
              token,
            );
          setNotice(
            "Graph changed. Review the regenerated preview before applying.",
          );
        } catch (conflict) {
          setError(errorText(conflict));
        }
      } else {
        // Preserve the same preview on a lost response so Retry cannot apply twice.
        setError(errorText(e));
      }
    } finally {
      loading.current = false;
      if (alive.current && request.current === token) setBusy(false);
    }
  }
  const pending = page?.coverage.resource_operation_incomplete;
  async function expandContext(key: string) {
    const token = request.current;
    const graph = await api.neighborhood(key, 1, pot, host);
    if (alive.current && token === request.current)
      setContext((previous) => mergeContext(previous, graph));
  }
  const enabled = Boolean(journal?.state?.rollback_enabled && !pending);
  if (outdated) {
    return (
      <PlanHistory
        key={`${host}:${pot}`}
        pot={pot}
        host={host}
        notice={
          "This host runs an older Potpie without commit history. Saved plans " +
          "show mutation status and scope; commits, comparisons and rollback " +
          "appear here once the host is updated."
        }
      />
    );
  }
  if (page?.coverage.legacy_only && page.headers.length === 0) {
    return <PlanHistory key={`${host}:${pot}`} pot={pot} host={host} />;
  }
  return (
    <section className="commits-view" aria-label="Mutation commits">
      <CommitHistory
        page={page}
        journal={journal}
        selected={selectedCommit}
        busy={busy}
        onSelect={inspect}
        onRefresh={refresh}
        onMore={loadMore}
      />
      <div className="commit-detail" aria-busy={busy}>
        <div className="commit-detail-content">
          {error && (
            <p className="commit-notice error-notice" role="alert">
              {error}
            </p>
          )}
          {notice && (
            <p className="commit-notice" role="status">
              {notice}
            </p>
          )}
          {page && !page.coverage.complete && !page.coverage.legacy_only && (
            <p className="commit-notice warning" role="status">
              History is incomplete. {page.coverage.indexing_lag} commits await
              indexing.
            </p>
          )}
          {pending && (
            <p className="commit-notice warning" role="status">
              A resource workflow is active or incomplete. Rollback is blocked
              until recovery completes.
            </p>
          )}
          {journal && !journal.state?.rollback_enabled && (
            <p className="commit-notice warning">
              Rollback is disabled for this pot.
            </p>
          )}
          {busy && (
            <p className="commit-progress" role="status">
              Loading commit data…
            </p>
          )}
          {!detail && !busy && (
            <div className="commit-empty">
              <CommitIcon name="commit" />
              <h2>Your graph, change by change</h2>
              <p>
                Select a commit to explore its records, compare fields, and
                review recovery options.
              </p>
            </div>
          )}
          {preview && detail && (
            <RollbackPreview
              result={preview}
              mode={previewMode}
              sequence={detail.header.sequence}
              busy={busy}
              onApply={apply}
              onCancel={() => setPreview(null)}
            />
          )}
          {detail && (
            <CommitInspector
              key={detail.header.commit_id}
              detail={detail}
              context={context}
              contextError={contextError}
              onExpandContext={expandContext}
              head={page?.coverage.head}
              enabled={enabled}
              busy={busy}
              selected={selectedRecord}
              onSelect={setSelectedRecord}
              onPreview={makePreview}
              onMore={moreDetail}
            />
          )}
        </div>
      </div>
    </section>
  );
}
