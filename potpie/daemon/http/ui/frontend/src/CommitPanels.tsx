import { useEffect, useRef, useState } from "react";
import CommitGraph from "./CommitGraph";
import type { GraphData } from "./types";
import CommitChanges from "./CommitChanges";
import {
  applyPreviewCommand,
  commitTitle,
  plural,
} from "./commitPresentation";
import type {
  CommitDetail,
  CommitHeader,
  CommitPage,
  JournalStatus,
  PreviewResult,
} from "./commitTypes";

const date = (value: string) =>
  new Date(value).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
const time = (value: string) =>
  new Date(value).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

export function CommitIcon({
  name,
}: {
  name: "commit" | "refresh" | "arrow" | "revert" | "search";
}) {
  const paths = {
    commit: "M2 10h4m8 0h4M14 10a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
    refresh: "M16 7a6.5 6.5 0 1 0 .5 5M16 3v4h-4",
    arrow: "m8 5 5 5-5 5",
    revert: "M3 8h8a5 5 0 0 1 0 10M3 8l5-5M3 8l5 5",
    search: "M13 13l4 4M14 8a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z",
  };
  return (
    <svg
      className="commit-icon"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}

export function CommitHistory({
  page,
  journal,
  selected,
  busy,
  onSelect,
  onRefresh,
  onMore,
}: {
  page: CommitPage | null;
  journal: JournalStatus | null;
  selected?: string;
  busy: boolean;
  onSelect: (row: CommitHeader) => void;
  onRefresh: () => void;
  onMore: () => void;
}) {
  const [query, setQuery] = useState("");
  const rows =
    page?.headers.filter((row) =>
      `${row.sequence} ${commitTitle(row)} ${row.message} ${row.origin} ${row.actor} ${row.commit_id}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
    ) ?? [];
  return (
    <aside className="commit-history" aria-label="Commit history">
      <div className="commit-history-head">
        <div className="commit-section-heading">
          <h1>Mutation history</h1>
          <button
            className="commit-button icon-only"
            onClick={onRefresh}
            disabled={busy}
            aria-label="Refresh history"
            title="Refresh history"
          >
            <CommitIcon name="refresh" />
          </button>
        </div>
        <p>Every change, with its context.</p>
        <label className="commit-filter">
          <CommitIcon name="search" />
          <input
            aria-label="Filter loaded commits"
            placeholder="Filter loaded commits…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      </div>
      <div className="commit-history-scroll" aria-busy={!page && busy}>
        {!page && (
          <p className="commit-empty-note">
            {busy
              ? "Loading commit history…"
              : "History could not be loaded. Try refreshing."}
          </p>
        )}
        {page?.coverage.legacy_only && (
          <p className="commit-empty-note">
            Journal coverage has not started. Earlier plans are available
            through graph history.
          </p>
        )}
        {page && !page.coverage.legacy_only && rows.length === 0 && (
          <p className="commit-empty-note">
            {query
              ? "No loaded commits match your search."
              : "No commits yet. Graph changes will appear here."}
          </p>
        )}
        {rows.map((row, index) => (
          <div key={row.commit_id}>
            {(index === 0 ||
              date(rows[index - 1].committed_at) !==
                date(row.committed_at)) && (
              <div className="commit-date">{date(row.committed_at)}</div>
            )}
            <button
              className={`commit-row${selected === row.commit_id ? " on" : ""}`}
              aria-current={selected === row.commit_id ? "true" : undefined}
              disabled={busy}
              onClick={() => onSelect(row)}
            >
              <span className="commit-rail">
                <CommitIcon name="commit" />
              </span>
              <span className="commit-row-body">
                <span className="commit-row-top">
                  <span className="commit-sequence">#{row.sequence}</span>
                  <span>{time(row.committed_at)}</span>
                  {row.commit_id === page?.coverage.head && (
                    <span className="commit-tag accent">Latest</span>
                  )}
                </span>
                <strong>{commitTitle(row)}</strong>
                <span className="commit-row-meta">
                  <span title={row.actor}>{row.actor}</span>
                  <span>{plural(row.affected_record_count, "record")}</span>
                </span>
                {!row.rollback_supported && (
                  <span
                    className="commit-barrier"
                    title={row.unsupported_reason || undefined}
                  >
                    Rollback barrier
                  </span>
                )}
              </span>
              <span className="commit-row-chevron">
                <CommitIcon name="arrow" />
              </span>
            </button>
          </div>
        ))}
        {page?.next_cursor && (
          <button
            className="commit-button commit-load-more"
            disabled={busy}
            onClick={onMore}
          >
            Load more commits
          </button>
        )}
      </div>
      {page && (
        <div className="commit-history-foot">
          <span
            className={`journal-indicator${page.coverage.complete ? " complete" : ""}`}
          />
          <span>
            {journal
              ? journal.capability.durable
                ? "Durable journal"
                : "Volatile journal"
              : "Journal status unavailable"}
          </span>
          <span>
            {page.coverage.complete ? "Up to date" : "Partial history"}
          </span>
        </div>
      )}
    </aside>
  );
}

export function CommitInspector({
  detail,
  context,
  contextError,
  onExpandContext,
  head,
  enabled,
  busy,
  selected,
  onSelect,
  onPreview,
  onMore,
}: {
  detail: CommitDetail;
  context: GraphData;
  contextError: string;
  onExpandContext: (key: string) => Promise<void>;
  head?: string | null;
  enabled: boolean;
  busy: boolean;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onPreview: (mode: "revert" | "rollback") => void;
  onMore: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const row = detail.header;
  return (
    <>
      <header className="commit-detail-head">
        <div className="commit-eyebrow">
          <CommitIcon name="commit" />
          <span>Commit #{row.sequence}</span>
          {head === row.commit_id && (
            <span className="commit-tag accent">Latest</span>
          )}
        </div>
        <h2>{commitTitle(row)}</h2>
        <p className="commit-byline commit-context-byline">
          <span>{row.actor}</span>
          <span>·</span>
          <time dateTime={row.committed_at}>
            {date(row.committed_at)} at {time(row.committed_at)}
          </time>
          <span className="commit-origin">{row.origin}</span>
        </p>
      </header>
      {!row.diff_complete && (
        <p className="commit-notice warning">
          Some field changes were not recorded. This diff is incomplete.
        </p>
      )}
      <CommitGraph
        detail={detail}
        context={context}
        contextError={contextError}
        onExpandContext={onExpandContext}
        selected={selected}
        onSelect={onSelect}
        busy={busy}
        onMore={onMore}
      />
      <details
        className="commit-secondary"
        open={showDetails}
        onToggle={(event) => setShowDetails(event.currentTarget.open)}
      >
        <summary>Details & recovery</summary>
        {showDetails && (
          <>
            <div className="commit-recovery">
              <div>
                <h3>Restore graph changes</h3>
                <p>
                  {!row.rollback_supported
                    ? row.unsupported_reason ||
                      "This commit cannot be reverted."
                    : "Review the changes before applying a revert or rollback."}
                </p>
              </div>
              <div className="commit-actions">
                <button
                  className="commit-button primary"
                  disabled={busy || !enabled || !row.rollback_supported}
                  onClick={() => onPreview("revert")}
                  title="Undo only this commit, preserving compatible later changes"
                >
                  <CommitIcon name="revert" />
                  Preview revert
                </button>
                <button
                  className="commit-button"
                  disabled={busy || !enabled || row.commit_id === head}
                  onClick={() => onPreview("rollback")}
                  title={
                    row.commit_id === head
                      ? "This is already the latest commit"
                      : "Undo the commits after this point"
                  }
                >
                  Roll back to here
                </button>
              </div>
            </div>
            <CommitChanges
              changes={detail.changes}
              context={detail.display_context}
              total={row.affected_record_count}
              partial={detail.next_offset !== null}
            />
            {detail.next_offset !== null && (
              <button
                className="commit-button commit-load-more"
                disabled={busy}
                onClick={onMore}
              >
                Load more changed records
              </button>
            )}
            <details className="commit-disclosure commit-technical">
              <summary>Commit metadata</summary>
              <dl>
                <dt>Message</dt>
                <dd>{row.message || "No message recorded"}</dd>
                <dt>Commit ID</dt>
                <dd>{row.commit_id}</dd>
                <dt>Required access</dt>
                <dd>{row.required_access}</dd>
              </dl>
            </details>
          </>
        )}
      </details>
    </>
  );
}

type CopyState = "idle" | "copied" | "manual";

/**
 * The browser session may inspect history and generate previews, which change
 * nothing. Applying one rewrites the graph, so it needs the daemon credential
 * and runs through the CLI's confirmed command instead of a button here.
 */
export function RollbackPreview({
  result,
  mode,
  sequence,
  pot,
  busy,
  onRefresh,
  onCancel,
}: {
  result: PreviewResult;
  mode: "revert" | "rollback";
  sequence: number;
  pot: string;
  busy: boolean;
  onRefresh: () => void;
  onCancel: () => void;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  const commandRef = useRef<HTMLElement>(null);
  const [copy, setCopy] = useState<CopyState>("idle");
  const command = applyPreviewCommand(result.preview.preview_id, pot);
  useEffect(() => {
    heading.current?.focus();
    setCopy("idle");
  }, [result.preview.preview_id]);
  // Without clipboard access, select the command so a manual copy is one
  // keystroke away.
  const selectCommand = () => {
    const element = commandRef.current;
    const selection =
      typeof window !== "undefined" ? window.getSelection() : null;
    if (!element || !selection) return;
    const range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
  };
  const copyCommand = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("unavailable");
      await navigator.clipboard.writeText(command);
      setCopy("copied");
    } catch {
      selectCommand();
      setCopy("manual");
    }
  };
  return (
    <section className="rollback-preview" aria-label="Rollback preview">
      <div className="commit-eyebrow">
        <CommitIcon name="revert" />
        Review before applying
      </div>
      <h2 ref={heading} tabIndex={-1}>
        {mode === "revert"
          ? `Revert commit #${sequence}`
          : `Roll back to commit #${sequence}`}
      </h2>
      <p>
        {mode === "revert"
          ? "Undo this commit while preserving compatible later changes."
          : "Undo all commits after this point."}{" "}
        Applying it creates a new commit that you can inspect and revert.
      </p>
      <div className="preview-summary">
        <strong>
          {plural(result.affected_record_count, "affected record")}
        </strong>
        <span>
          Expires {date(result.preview.expires_at)} at{" "}
          {time(result.preview.expires_at)}
        </span>
      </div>
      {result.changes_truncated && (
        <p className="commit-notice warning">
          Only part of this preview is shown. All changes are validated before
          applying.
        </p>
      )}
      <CommitChanges
        changes={result.changes}
        total={result.affected_record_count}
        partial={result.changes_truncated}
      />
      <details className="commit-disclosure">
        <summary>Validation details</summary>
        <dl>
          <dt>Preview ID</dt>
          <dd>{result.preview.preview_id}</dd>
          <dt>Expected HEAD</dt>
          <dd>{result.preview.expected_head}</dd>
          <dt>Required access</dt>
          <dd>{result.preview.required_access}</dd>
          <dt>Atomic limits</dt>
          <dd>
            {result.limits.max_records.toLocaleString()} records ·{" "}
            {result.limits.max_bytes.toLocaleString()} bytes
          </dd>
        </dl>
      </details>
      <div className="preview-apply">
        <h3>Apply from your terminal</h3>
        <p>
          This page can preview changes but not apply them. Applying needs the
          Potpie CLI, so run this command before the preview expires on{" "}
          {date(result.preview.expires_at)} at{" "}
          {time(result.preview.expires_at)}, then refresh the history.
        </p>
        <pre className="preview-command" aria-label="Command to apply this preview">
          <code ref={commandRef}>{command}</code>
        </pre>
        {copy !== "idle" && (
          <p className="preview-copy-status" role="status">
            {copy === "copied"
              ? "Command copied."
              : "Copying is unavailable here. The command is selected, so copy it manually."}
          </p>
        )}
      </div>
      <div className="commit-actions">
        <button className="commit-button primary" onClick={copyCommand}>
          Copy command
        </button>
        <button className="commit-button" disabled={busy} onClick={onRefresh}>
          <CommitIcon name="refresh" />
          Refresh history
        </button>
        <button className="commit-button" disabled={busy} onClick={onCancel}>
          Cancel preview
        </button>
      </div>
    </section>
  );
}
