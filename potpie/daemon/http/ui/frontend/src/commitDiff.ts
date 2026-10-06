import type {
  CommitDetail,
  FieldValue,
  RecordChange,
  RecordedContext,
} from "./commitTypes";
import type { GraphData, GraphEdge, GraphNode } from "./types";

export function fieldText(value: FieldValue): string {
  if (!value.present) return "(absent)";
  return JSON.stringify(value.value, null, 2) ?? "(unavailable)";
}

export function changeStatus(change: RecordChange): string {
  if (change.action === "create") return "added";
  if (change.action === "retire") return "retired";
  if (change.action === "restore") return "reactivated";
  if (
    change.fields.some(
      (f) =>
        ["invalid_at", "valid_until"].includes(
          f.path[f.path.length - 1] || "",
        ) &&
        f.after.present &&
        f.after.value !== null,
    )
  )
    return "invalidated";
  return "modified";
}

export type GraphPhase = "before" | "changes" | "after";

/** Read only recorded values. A missing value differs from an unrecorded one. */
function recordedValue(
  effects: RecordChange[],
  path: string[],
  phase: "before" | "after",
): FieldValue | undefined {
  const ordered = phase === "before" ? effects : [...effects].reverse();
  for (const effect of ordered) {
    for (const field of effect.fields) {
      if (
        !field.path.every((part, index) => path[index] === part) ||
        field.path.length > path.length
      )
        continue;
      let value = field[phase];
      for (const key of path.slice(field.path.length))
        value = childValue(value.value, key);
      return value;
    }
    const snapshot =
      phase === "before" ? effect.before_record : effect.after_record;
    if (snapshot) {
      let value: FieldValue = { present: true, value: snapshot.fields };
      for (const key of path) value = childValue(value.value, key);
      return value;
    }
  }
}

function childValue(value: unknown, key: string): FieldValue {
  if (
    value &&
    typeof value === "object" &&
    Object.prototype.hasOwnProperty.call(value, key)
  )
    return { present: true, value: (value as Record<string, unknown>)[key] };
  return { present: false, value: null };
}

export function recordInPhase(
  effects: RecordChange[],
  phase: GraphPhase,
): boolean {
  if (phase === "changes" || !effects.length) return true;
  if (phase === "before" && effects[0].action === "create") return false;
  if (phase === "after" && effects[0].action === "retire") return false;
  if (phase === "before" && effects[0].action === "restore") return false;
  return recordedValue(effects, ["active"], phase)?.value !== false;
}

/** An affected subgraph, never a reconstruction from today's live graph. */
export function diffGraph(
  detail: CommitDetail,
  phase: GraphPhase = "changes",
  recordIds?: Set<string>,
): GraphData {
  const changes = new Map<string, RecordChange[]>();
  for (const change of detail.changes) {
    const effects = changes.get(change.record_id) || [];
    effects.push(change);
    changes.set(change.record_id, effects);
  }
  const contexts = new Map(detail.display_context.map((c) => [c.record_id, c]));
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const side = phase === "before" ? "before" : "after";
  function value(
    id: string,
    path: string[],
    fallback: unknown,
    snapshotSide: "before" | "after" = side,
  ): unknown {
    const recorded = recordedValue(changes.get(id) || [], path, snapshotSide);
    return recorded
      ? recorded.present
        ? recorded.value
        : undefined
      : fallback;
  }
  function node(id: string): GraphNode | undefined {
    const effects = changes.get(id) || [];
    if (!recordInPhase(effects, phase)) return;
    const old = nodes.get(id);
    if (old) return old;
    const context = contexts.get(id);
    const change = effects[0];
    const status = change ? changeStatus(change) : "recorded context";
    const key = context?.logical_key || change?.logical_key || id;
    const snapshotSide =
      phase === "changes" && status === "retired" ? "before" : side;
    const name = value(id, ["properties", "name"], context?.name, snapshotSide);
    const recordedLabels = value(id, ["labels"], context?.labels, snapshotSide);
    const labels = Array.isArray(recordedLabels)
      ? recordedLabels.filter(
          (label): label is string => typeof label === "string",
        )
      : [];
    const created: GraphNode = {
      id,
      key,
      labels,
      type: labels[0] || "Entity",
      caption: typeof name === "string" && name ? name : key,
      properties: {},
      diff_status: phase === "before" && change ? "before" : status,
      ghost:
        phase !== "before" &&
        (status === "retired" || status === "invalidated"),
    };
    nodes.set(id, created);
    return created;
  }
  function relationship(
    id: string,
    context: RecordedContext | undefined,
    snapshotSide: "before" | "after",
  ) {
    const source = value(
      id,
      ["subject_record_id"],
      context?.endpoint_record_ids[0],
      snapshotSide,
    );
    const target = value(
      id,
      ["object_record_id"],
      context?.endpoint_record_ids[1],
      snapshotSide,
    );
    const predicate = value(
      id,
      ["predicate"],
      context?.predicate,
      snapshotSide,
    );
    if (typeof source !== "string" || typeof target !== "string") return;
    return {
      source,
      target,
      predicate: typeof predicate === "string" ? predicate : "recorded claim",
    };
  }
  for (const [id, effects] of changes) {
    if (recordIds && !recordIds.has(id)) continue;
    const change = effects[0];
    if (!recordInPhase(effects, phase)) continue;
    if (change.kind === "entity") node(id);
    else {
      const context = contexts.get(id);
      const status = changeStatus(change);
      const current = relationship(
        id,
        context,
        phase === "changes" && status === "retired" ? "before" : side,
      );
      if (!current) continue;
      const add = (
        edge: typeof current,
        diffStatus: string,
        variant?: string,
      ) => {
        if (!node(edge.source) || !node(edge.target)) return;
        edges.push({
          id: variant || id,
          record_id: id,
          ...edge,
          diff_status: diffStatus,
        });
      };
      const previous = relationship(id, context, "before");
      if (
        phase === "changes" &&
        recordInPhase(effects, "before") &&
        previous &&
        JSON.stringify(previous) !== JSON.stringify(current)
      )
        add(previous, "previous", `${id}:before`);
      add(current, phase === "before" ? "before" : status);
    }
  }
  return { nodes: [...nodes.values()], edges };
}
