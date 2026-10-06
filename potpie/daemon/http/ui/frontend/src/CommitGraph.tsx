import { useMemo, useState } from "react";
import GraphView from "./GraphView";
import { groupRecords, plural, recordStatus } from "./commitPresentation";
import { mutationInContext } from "./mutationContext";
import MutationSelection from "./MutationSelection";
import type { CommitDetail } from "./commitTypes";
import type { GraphData } from "./types";
import "./commitGraph.css";

export default function CommitGraph({
  detail,
  context,
  contextError,
  selected,
  onSelect,
  onExpandContext,
  busy,
  onMore,
}: {
  detail: CommitDetail;
  context: GraphData;
  contextError: string;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onExpandContext: (key: string) => Promise<void>;
  busy: boolean;
  onMore: () => void;
}) {
  const [expanded, setExpanded] = useState<string[]>([]);
  const [expanding, setExpanding] = useState(false);
  const [expandError, setExpandError] = useState("");
  const records = useMemo(() => groupRecords(detail.changes), [detail.changes]);
  const graph = useMemo(
    () => mutationInContext(detail, context, expanded),
    [detail, context, expanded],
  );
  const counts = useMemo(() => {
    const result: Record<string, number> = {};
    for (const effects of records) {
      const state = recordStatus(effects);
      result[state] = (result[state] || 0) + 1;
    }
    return result;
  }, [records]);
  const selectedNode = graph.nodes.find((node) => node.id === selected);
  const selectedEdge =
    graph.edges.find((edge) => edge.id === selected) ||
    graph.edges.find((edge) => edge.record_id === selected);
  const selectedEffects = records.find(
    (effects) => effects[0].record_id === selected,
  );
  const surrounding = graph.nodes.filter(
    (node) => node.diff_status === "context",
  ).length;
  async function expand() {
    if (!selectedNode || expanding) return;
    const node = selectedNode;
    setExpanding(true);
    setExpandError("");
    try {
      await onExpandContext(node.key);
      setExpanded((previous) => [...new Set([...previous, node.id])]);
    } catch (error) {
      setExpandError(error instanceof Error ? error.message : String(error));
    } finally {
      setExpanding(false);
    }
  }
  return (
    <section className="mutation-context" aria-label="Mutation in context">
      <div className="mutation-context-heading">
        <div>
          <h3>Mutation in context</h3>
          <p>Follow the highlighted changes through the surrounding graph.</p>
        </div>
        <span>{plural(records.length, "change")}</span>
      </div>
      <div
        className="mutation-context-stage"
        role="group"
        aria-label="Interactive graph. Use arrow keys to inspect changes and Escape to close details."
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.target !== event.currentTarget) return;
          if (event.key === "Escape") {
            onSelect(null);
            return;
          }
          if (
            !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(
              event.key,
            ) ||
            !records.length
          )
            return;
          event.preventDefault();
          const index = records.findIndex(
            (effects) => effects[0].record_id === selected,
          );
          const step =
            event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
          onSelect(
            records[(index + step + records.length) % records.length][0]
              .record_id,
          );
        }}
      >
        <GraphView
          fitOnLoad
          focusOnSelect
          data={graph}
          selectedId={selected}
          onSelect={(node) => onSelect(node?.id || null)}
          onSelectEdge={(edge) => onSelect(edge.record_id || edge.id)}
          onExpand={(node) => onSelect(node.id)}
        />
        {graph.nodes.length === 0 && (
          <p className="mutation-context-empty">
            No graph connections were recorded. Open details below to inspect
            the changes.
          </p>
        )}
        <div className="mutation-context-legend" aria-label="Change legend">
          {Object.entries(counts).map(([state, count]) => (
            <span className={`change-status ${state}`} key={state}>
              <i />
              {count} {state === "modified" ? "updated" : state}
            </span>
          ))}
          <span className="context">
            <i />
            {surrounding} surrounding
          </span>
        </div>
        {!selected && (
          <span className="mutation-context-hint">
            Click a node or connection to dig in
          </span>
        )}
        {selected && (
          <MutationSelection
            key={selected}
            effects={selectedEffects}
            node={selectedNode}
            edge={selectedEdge}
            graph={graph}
            detail={detail}
            onClose={() => onSelect(null)}
            onExpand={expand}
            expanding={expanding}
            error={expandError}
          />
        )}
      </div>
      <div className="mutation-context-foot">
        <span title="The highlighted records come from this commit. Gray nodes and connections come from the current graph, which may include later changes.">
          Changes from this commit · current surrounding graph
        </span>
        {contextError ? (
          <span role="status">
            Surrounding graph unavailable. Refresh history to retry.
          </span>
        ) : graph.truncated ? (
          <span>Some surrounding connections are outside this view.</span>
        ) : null}
        {detail.next_offset !== null && (
          <button className="commit-button" disabled={busy} onClick={onMore}>
            Load remaining changes
          </button>
        )}
      </div>
    </section>
  );
}
