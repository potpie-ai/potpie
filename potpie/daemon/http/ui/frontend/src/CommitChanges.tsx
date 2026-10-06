import { useMemo, useState } from "react";
import { changeStatus, fieldText } from "./commitDiff";
import {
  compactChanges,
  filterChanges,
  groupRecords,
  plural,
  recordFields,
  recordStatus,
  recordTitle,
  statusMarks,
} from "./commitPresentation";
import type { RecordChange, RecordedContext } from "./commitTypes";

const PAGE_SIZE = 25;
const GROUP_LABELS: Record<string, string> = {
  entity: "Entities",
  claim: "Relationships",
  relation: "Relationships",
};
const NO_CONTEXT: RecordedContext[] = [];

export default function CommitChanges({
  changes,
  context = NO_CONTEXT,
  selected = null,
  total,
  partial = false,
}: {
  changes: RecordChange[];
  context?: RecordedContext[];
  selected?: string | null;
  total: number;
  partial?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [limit, setLimit] = useState(PAGE_SIZE);
  const contexts = useMemo(
    () => new Map(context.map((item) => [item.record_id, item])),
    [context],
  );
  const compact = compactChanges(changes, total, partial);
  const records = groupRecords(changes);
  const counts: Record<string, number> = {};
  for (const effects of records) {
    const state = recordStatus(effects);
    counts[state] = (counts[state] || 0) + 1;
  }
  const matches = new Set(
    filterChanges(changes, contexts, query, "all").map(
      (change) => change.record_id,
    ),
  );
  const filtered = records.filter((effects) =>
    selected
      ? effects[0].record_id === selected
      : matches.has(effects[0].record_id) &&
        (status === "all" || recordStatus(effects) === status),
  );
  const shown = filtered.slice(0, limit);
  const groups = new Map<string, RecordChange[][]>();
  for (const effects of shown) {
    const group = GROUP_LABELS[effects[0].kind] || "Other records";
    groups.set(group, [...(groups.get(group) || []), effects]);
  }
  const resetFilters = () => {
    setQuery("");
    setStatus("all");
    setLimit(PAGE_SIZE);
  };
  return (
    <div className="change-fields">
      {!compact && !selected && (
        <div className="change-explorer">
          <div className="change-explorer-heading">
            <h3>Change breakdown</h3>
            <span>
              {partial
                ? `${records.length.toLocaleString()} of ${plural(total, "record")} loaded`
                : plural(records.length, "record")}
            </span>
          </div>
          <div className="change-impact-bar" aria-hidden="true">
            {Object.entries(counts).map(([state, count]) => (
              <span key={state} className={state} style={{ flexGrow: count }} />
            ))}
          </div>
          <div
            className="change-filters"
            role="group"
            aria-label="Filter changes by status"
          >
            <button
              aria-pressed={status === "all"}
              onClick={() => {
                setStatus("all");
                setLimit(PAGE_SIZE);
              }}
            >
              All <b>{records.length}</b>
            </button>
            {Object.entries(counts).map(([state, count]) => (
              <button
                key={state}
                aria-pressed={status === state}
                onClick={() => {
                  setStatus(state);
                  setLimit(PAGE_SIZE);
                }}
              >
                <span className={`change-status ${state}`}>
                  {statusMarks[state]} {state}
                </span>
                <b>{count}</b>
              </button>
            ))}
          </div>
          <input
            className="change-search"
            aria-label="Filter loaded changed records"
            placeholder="Find a record or field…"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setLimit(PAGE_SIZE);
            }}
          />
          {(query || status !== "all" || partial) && (
            <p className="change-filter-note" role="status">
              {plural(filtered.length, "matching record")} in the loaded
              changes.
              {partial && " Load more records below to extend the search."}
            </p>
          )}
        </div>
      )}
      {filtered.length === 0 && (
        <div className="commit-empty-note">
          <p>
            {selected
              ? "This endpoint provides context. It has no changed fields in this commit."
              : changes.length
                ? "No records match these filters."
                : "No field changes were recorded."}
          </p>
          {!selected && changes.length > 0 && (
            <button className="commit-button" onClick={resetFilters}>
              Clear filters
            </button>
          )}
        </div>
      )}
      {[...groups].map(([group, records]) => (
        <section className="change-group" key={group} aria-label={group}>
          {!compact && !selected && (
            <h4 className="change-group-heading">
              {group}
              <span>{records.length}</span>
            </h4>
          )}
          {records.map((effects) => (
            <ChangeRecord
              key={effects[0].record_id}
              effects={effects}
              name={recordTitle(effects[0], contexts)}
              expanded={Boolean(selected) || compact}
            />
          ))}
        </section>
      ))}
      {filtered.length > shown.length && (
        <button
          className="commit-button commit-load-more"
          onClick={() => setLimit((previous) => previous + PAGE_SIZE)}
        >
          Show {Math.min(PAGE_SIZE, filtered.length - shown.length)} more ·{" "}
          {shown.length} of {filtered.length} shown
        </button>
      )}
    </div>
  );
}

function ChangeRecord({
  effects,
  name,
  expanded,
}: {
  effects: RecordChange[];
  name: string;
  expanded: boolean;
}) {
  const [openOverride, setOpen] = useState<boolean | null>(null);
  const open = openOverride ?? expanded;
  const change = effects[0];
  const status = recordStatus(effects);
  const fields = new Set(
    effects.flatMap((effect) =>
      effect.fields.map((field) => JSON.stringify(field.path)),
    ),
  );
  return (
    <details
      className="change-record"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span className={`change-mark ${status}`}>
          {statusMarks[status] || "~"}
        </span>
        <span className="change-record-title">
          <strong>{name}</strong>
          <span>
            {name !== change.logical_key ? change.logical_key : change.kind}
            {fields.size > 0 && ` · ${plural(fields.size, "changed field")}`}
          </span>
        </span>
        <span className={`change-status ${status}`}>{status}</span>
        <svg
          className="commit-icon"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          aria-hidden="true"
        >
          <path d="m8 5 5 5-5 5" />
        </svg>
      </summary>
      {open && (
        <div>
          <RecordDiff change={change} />
          {effects.length > 1 && (
            <details className="change-additional">
              <summary>
                Additional recorded changes · {effects.length - 1}
              </summary>
              {effects.slice(1).map((effect, index) => (
                <div key={index}>
                  <div className="change-effect-label">
                    Change {index + 2} of {effects.length}{" "}
                    <span className={`change-status ${changeStatus(effect)}`}>
                      {changeStatus(effect)}
                    </span>
                  </div>
                  <RecordDiff change={effect} />
                </div>
              ))}
            </details>
          )}
          <div className="change-record-id">
            <span>Record ID</span>
            <code>{change.record_id}</code>
          </div>
        </div>
      )}
    </details>
  );
}

function RecordDiff({ change }: { change: RecordChange }) {
  const [showAll, setShowAll] = useState(false);
  const fields = recordFields(change);
  const shown = showAll ? fields : fields.slice(0, 12);
  return (
    <div className="change-record-body">
      {fields.length > 0 ? (
        <div className="change-table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Field</th>
                <th scope="col">
                  <span className="diff-sign">−</span> Before
                </th>
                <th scope="col">
                  <span className="diff-sign">+</span> After
                </th>
              </tr>
            </thead>
            <tbody>
              {shown.map((field) => (
                <tr key={JSON.stringify(field.path)}>
                  <th scope="row" title={field.path.join(".")}>
                    {(field.path[0] === "properties"
                      ? field.path.slice(1)
                      : field.path
                    ).join(".")}
                  </th>
                  <td className="field-before">
                    <pre>{fieldText(field.before)}</pre>
                  </td>
                  <td className="field-after">
                    <pre>{fieldText(field.after)}</pre>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="commit-empty-note">
          No field-level details were recorded for this change.
        </p>
      )}
      {fields.length > shown.length && (
        <button
          className="commit-button commit-load-more"
          onClick={() => setShowAll(true)}
        >
          Show all {fields.length} fields
        </button>
      )}
    </div>
  );
}
