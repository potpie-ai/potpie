import { changeStatus } from "./commitDiff.ts";
import type {
  CommitHeader,
  RecordChange,
  RecordedContext,
} from "./commitTypes";

export const statusMarks: Record<string, string> = {
  added: "+",
  modified: "~",
  invalidated: "⊘",
  retired: "−",
  reactivated: "↺",
};
export const plural = (count: number, noun: string) =>
  `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;

export function groupRecords(changes: RecordChange[]): RecordChange[][] {
  const records = new Map<string, RecordChange[]>();
  for (const change of changes) {
    const effects = records.get(change.record_id);
    if (effects) effects.push(change);
    else records.set(change.record_id, [change]);
  }
  return [...records.values()];
}

// The receipt orders semantic changes before audit stamps. The first effect
// supplies the same lifecycle status used by the recorded graph.
export const recordStatus = (effects: RecordChange[]) =>
  changeStatus(effects[0]);

/** Quote an argument for a POSIX shell only when it needs it. */
const shellArg = (value: string) =>
  /^[A-Za-z0-9._:@/=+-]+$/.test(value)
    ? value
    : `'${value.replace(/'/g, `'\\''`)}'`;

/** The CLI command that applies a preview. The explorer only generates
 * previews; applying one needs the daemon credential the CLI holds. */
export function applyPreviewCommand(previewId: string, pot: string): string {
  return `potpie graph apply-preview ${shellArg(previewId)} --pot ${shellArg(pot)} --yes`;
}

export function commitTitle(row: CommitHeader): string {
  if (/^revert [a-f0-9]{32}$/i.test(row.message)) return "Revert graph changes";
  return (
    row.message ||
    ({ graph: "Graph update", resource: "Resource update" }[row.origin] ??
      row.origin)
  );
}

export function recordTitle(
  change: RecordChange,
  context: Map<string, RecordedContext>,
): string {
  const item = context.get(change.record_id);
  const snapshot = change.after_record?.fields || change.before_record?.fields;
  const predicate = item?.predicate || snapshot?.predicate;
  const endpoints = item?.endpoint_record_ids.length
    ? item.endpoint_record_ids
    : [snapshot?.subject_record_id, snapshot?.object_record_id];
  if (
    typeof predicate === "string" &&
    endpoints.length === 2 &&
    endpoints.every((id) => typeof id === "string")
  ) {
    const [source, target] = (endpoints as string[]).map(
      (id) => context.get(id)?.name || context.get(id)?.logical_key || id,
    );
    return `${source} → ${predicate} → ${target}`;
  }
  if (item?.name) return item.name;
  return change.logical_key;
}

export function recordFields(change: RecordChange): RecordChange["fields"] {
  if (change.fields.length) return change.fields;
  const before = snapshotFields(change.before_record?.fields ?? {});
  const after = snapshotFields(change.after_record?.fields ?? {});
  return [...new Set([...before.keys(), ...after.keys()])]
    .sort()
    .map((key) => ({
      path: JSON.parse(key) as string[],
      before: { present: before.has(key), value: before.get(key) },
      after: { present: after.has(key), value: after.get(key) },
    }));
}

function snapshotFields(fields: Record<string, unknown>): Map<string, unknown> {
  const result = new Map<string, unknown>();
  function visit(value: unknown, path: string[]) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      Object.keys(value).length
    ) {
      for (const [key, item] of Object.entries(value))
        visit(item, [...path, key]);
    } else result.set(JSON.stringify(path), value);
  }
  for (const [key, value] of Object.entries(fields)) visit(value, [key]);
  return result;
}

// Small edits can be read in place. Larger or partially loaded changes start
// collapsed so expanding one record never produces hundreds of diff tables.
export function compactChanges(
  changes: RecordChange[],
  total: number,
  partial: boolean,
): boolean {
  return (
    !partial &&
    total <= 3 &&
    changes.reduce((count, change) => count + recordFields(change).length, 0) <=
      12
  );
}

export function filterChanges(
  changes: RecordChange[],
  context: Map<string, RecordedContext>,
  query: string,
  status: string,
): RecordChange[] {
  const search = query.trim().toLowerCase();
  return changes.filter(
    (change) =>
      (status === "all" || changeStatus(change) === status) &&
      `${recordTitle(change, context)} ${change.logical_key} ${change.record_id} ${change.kind} ${recordFields(
        change,
      )
        .map((field) => field.path.join("."))
        .join(" ")}`
        .toLowerCase()
        .includes(search),
  );
}
