import { useState } from "react";
import CommitChanges from "./CommitChanges";
import { fieldText } from "./commitDiff";
import { recordFields, recordStatus } from "./commitPresentation";
import { endpointId } from "./mutationContext";
import type { CommitDetail, FieldValue, RecordChange } from "./commitTypes";
import type { GraphData, GraphEdge, GraphNode } from "./types";

const INTERNAL =
  /^(prov_|record_id$|entity_key$|claim_key$|group_id$|created_|updated_|actor_|mutation_|graph_contract_|idempotency_|plan_id$|uuid$)/;
const DETAIL_ONLY = new Set([
  "active",
  "labels",
  "subject_record_id",
  "object_record_id",
  "subject_key",
  "object_key",
  "predicate",
  "confidence",
  "evidence",
  "evidence_strength",
  "truth",
  "subgraph",
  "source_ref",
  "origin",
  "authority",
]);
const FIELD_PRIORITY = new Map(
  ["name", "title", "summary", "description", "fact", "status", "state"].map(
    (name, index) => [name, index],
  ),
);
const DESCRIPTION_FIELDS = new Set(["summary", "description", "fact"]);
function readableFields(change: RecordChange) {
  const descriptive = new Set<string>();
  return recordFields(change)
    .filter((field) => {
      const name =
        (field.path[0] === "properties" ? field.path[1] : field.path[0]) || "";
      if (change.action === "create" && !FIELD_PRIORITY.has(name)) return false;
      return !INTERNAL.test(name) && !DETAIL_ONLY.has(name);
    })
    .sort((left, right) => {
      const rank = (path: string[]) =>
        FIELD_PRIORITY.get(path[path.length - 1]) ?? FIELD_PRIORITY.size;
      return rank(left.path) - rank(right.path);
    })
    .filter((field) => {
      const name = field.path[field.path.length - 1];
      if (change.action === "create" && name === "name") return false;
      if (!DESCRIPTION_FIELDS.has(name)) return true;
      const content = JSON.stringify([field.before, field.after]);
      if (descriptive.has(content)) return false;
      descriptive.add(content);
      return true;
    });
}

const conciseValue = (field: FieldValue) =>
  typeof field.value === "string" ? field.value : fieldText(field);

export default function MutationSelection({
  effects,
  node,
  edge,
  graph,
  detail,
  onClose,
  onExpand,
  expanding,
  error,
}: {
  effects?: RecordChange[];
  node?: GraphNode;
  edge?: GraphEdge;
  graph: GraphData;
  detail: CommitDetail;
  onClose: () => void;
  onExpand: () => void;
  expanding: boolean;
  error: string;
}) {
  const [showAll, setShowAll] = useState(false);
  const status = effects ? recordStatus(effects) : "context";
  const name = (id: string) =>
    graph.nodes.find((item) => item.id === id)?.caption || id;
  const title =
    node?.caption ||
    (edge ? edge.predicate : effects?.[0].logical_key || "Record");
  const fields = effects ? readableFields(effects[0]) : [];
  return (
    <aside className="mutation-selection" aria-label="Selected graph item">
      <header>
        <span className={`change-status ${status}`}>
          {status === "context"
            ? "Surrounding graph"
            : status === "modified"
              ? "Updated in this commit"
              : `${status} in this commit`}
        </span>
        <button aria-label="Close graph details" onClick={onClose}>
          ×
        </button>
      </header>
      <div className="mutation-selection-body">
        <h3>{title}</h3>
        {edge && (
          <p className="mutation-selection-endpoints">
            {name(endpointId(edge.source))}
            <span>↓ {edge.predicate}</span>
            {name(endpointId(edge.target))}
          </p>
        )}
        {effects ? (
          <>
            {fields.length ? (
              <div className="mutation-value-diffs">
                {fields.slice(0, 3).map((field) => (
                  <div key={JSON.stringify(field.path)}>
                    <strong>
                      {(field.path[0] === "properties"
                        ? field.path.slice(1)
                        : field.path
                      ).join(".")}
                    </strong>
                    {field.before.present && (
                      <pre className="before">
                        <span>− </span>
                        {conciseValue(field.before)}
                      </pre>
                    )}
                    {field.after.present && (
                      <pre className="after">
                        <span>+ </span>
                        {conciseValue(field.after)}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p>
                {status === "added"
                  ? "Added to the graph by this commit."
                  : status === "retired"
                    ? "Removed from the graph by this commit."
                    : "Recorded metadata or connections changed."}
              </p>
            )}
            <details
              className="mutation-selection-full"
              open={showAll}
              onToggle={(event) => setShowAll(event.currentTarget.open)}
            >
              <summary>All recorded fields</summary>
              {showAll && (
                <CommitChanges
                  changes={effects}
                  context={detail.display_context}
                  selected={effects[0].record_id}
                  total={1}
                />
              )}
            </details>
          </>
        ) : (
          <p>
            {node?.summary || "Connected to the mutation in the current graph."}
          </p>
        )}
        {node && (
          <button
            className="commit-button"
            onClick={onExpand}
            disabled={expanding}
          >
            {expanding ? "Loading connections…" : "Explore connections"}
          </button>
        )}
        {error && <p role="alert">{error}</p>}
      </div>
    </aside>
  );
}
