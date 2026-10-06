import test from "node:test";
import assert from "node:assert/strict";
import {
  diffGraph,
  fieldText,
  changeStatus,
  recordInPhase,
} from "../src/commitDiff.ts";
import type { CommitDetail, RecordChange } from "../src/commitTypes.ts";

const patch = (id: string): RecordChange => ({
  record_id: id,
  logical_key: "claim:parallel",
  kind: "claim",
  action: "patch",
  fields: [],
  before_record: null,
  after_record: null,
});
const detail = {
  changes: [patch("edge-a"), patch("edge-b")],
  display_context: [
    {
      record_id: "source",
      logical_key: "service:a",
      kind: "entity",
      labels: ["Service"],
      name: "Source",
      endpoint_record_ids: [],
    },
    {
      record_id: "target",
      logical_key: "service:b",
      kind: "entity",
      labels: ["Service"],
      name: "Target",
      endpoint_record_ids: [],
    },
    {
      record_id: "edge-a",
      logical_key: "claim:parallel",
      kind: "claim",
      labels: [],
      name: "",
      predicate: "DEPENDS_ON",
      endpoint_record_ids: ["source", "target"],
    },
    {
      record_id: "edge-b",
      logical_key: "claim:parallel",
      kind: "claim",
      labels: [],
      name: "",
      predicate: "DEPENDS_ON",
      endpoint_record_ids: ["source", "target"],
    },
  ],
} as CommitDetail;

test("record incarnation IDs preserve parallel claims", () => {
  const graph = diffGraph(detail);
  assert.deepEqual(
    graph.edges.map((edge) => edge.id),
    ["edge-a", "edge-b"],
  );
  assert.equal(graph.nodes.length, 2);
  assert.ok(
    graph.nodes.every((node) => node.diff_status === "recorded context"),
  );
  assert.deepEqual(
    graph.nodes.map((node) => node.caption),
    ["Source", "Target"],
  );
});

test("before and after include the correct lifecycle records and never dangling edges", () => {
  const added = {
    ...patch("new"),
    kind: "entity",
    action: "create",
    after_record: {
      fields: { properties: { name: "New service" }, labels: ["Service"] },
    },
  };
  const retired = { ...patch("target"), kind: "entity", action: "retire" };
  const restored = { ...patch("restored"), kind: "entity", action: "restore" };
  const receipt = {
    ...detail,
    changes: [...detail.changes, added, retired, restored],
  };
  const before = diffGraph(receipt, "before");
  const after = diffGraph(receipt, "after");
  assert.ok(before.nodes.some((node) => node.id === "target"));
  assert.ok(
    !before.nodes.some((node) => ["new", "restored"].includes(node.id)),
  );
  assert.ok(after.nodes.some((node) => node.id === "new"));
  assert.ok(after.nodes.some((node) => node.id === "restored"));
  assert.ok(!after.nodes.some((node) => node.id === "target"));
  assert.equal(after.edges.length, 0);
  assert.equal(diffGraph(receipt).edges.length, 2);
});

test("before uses old names and labels even though display context captures after", () => {
  const renamed = {
    ...patch("source"),
    kind: "entity",
    fields: [
      {
        path: ["properties", "name"],
        before: { present: true, value: "Original name" },
        after: { present: true, value: "New name" },
      },
      {
        path: ["labels"],
        before: { present: true, value: ["Team"] },
        after: { present: true, value: ["Service"] },
      },
    ],
  };
  const receipt = { ...detail, changes: [renamed] };
  assert.equal(diffGraph(receipt, "before").nodes[0].caption, "Original name");
  assert.equal(diffGraph(receipt, "before").nodes[0].type, "Team");
  assert.equal(diffGraph(receipt, "after").nodes[0].caption, "New name");
  renamed.fields[0].before = { present: false, value: null };
  assert.equal(diffGraph(receipt, "before").nodes[0].caption, "service:a");
});

test("rewired relationships show old and new paths in the overlay, and exact paths per side", () => {
  const rewired = {
    ...patch("edge-a"),
    fields: [
      {
        path: ["object_record_id"],
        before: { present: true, value: "old-target" },
        after: { present: true, value: "target" },
      },
      {
        path: ["predicate"],
        before: { present: true, value: "USES" },
        after: { present: true, value: "DEPENDS_ON" },
      },
    ],
  };
  const receipt = { ...detail, changes: [rewired] };
  const overlay = diffGraph(receipt);
  assert.equal(overlay.edges.length, 2);
  assert.deepEqual(
    overlay.edges.map((edge) => edge.record_id),
    ["edge-a", "edge-a"],
  );
  assert.equal(new Set(overlay.edges.map((edge) => edge.id)).size, 2);
  assert.deepEqual(
    overlay.edges.map((edge) => edge.target),
    ["old-target", "target"],
  );
  assert.equal(diffGraph(receipt, "before").edges[0].predicate, "USES");
  assert.equal(diffGraph(receipt, "after").edges[0].target, "target");
});

test("snapshots and parent-object field changes supply historical values without mutation", () => {
  const created = {
    ...patch("source"),
    kind: "entity",
    action: "create",
    after_record: {
      fields: { properties: { name: "Created" }, labels: ["Service"] },
    },
  };
  const audit = {
    ...patch("source"),
    kind: "entity",
    fields: [
      {
        path: ["properties", "actor"],
        before: { present: false, value: null },
        after: { present: true, value: "agent" },
      },
    ],
  };
  const receipt = { ...detail, changes: [created, audit] };
  const saved = JSON.stringify(receipt);
  assert.equal(diffGraph(receipt, "after").nodes[0].caption, "Created");
  assert.equal(diffGraph(receipt, "before").nodes.length, 0);
  assert.equal(JSON.stringify(receipt), saved);
  const parent = {
    ...audit,
    fields: [
      {
        path: ["properties"],
        before: { present: true, value: { name: "Old" } },
        after: { present: true, value: {} },
      },
    ],
  };
  assert.equal(
    diffGraph({ ...detail, changes: [parent] }, "before").nodes[0].caption,
    "Old",
  );
  assert.equal(
    diffGraph({ ...detail, changes: [parent] }, "after").nodes[0].caption,
    "service:a",
  );
});

test("invalidated claims remain inspectable while inactive records are omitted", () => {
  const invalidated = {
    ...patch("edge-a"),
    fields: [
      {
        path: ["invalid_at"],
        before: { present: false, value: null },
        after: { present: true, value: "2026-10-01" },
      },
    ],
  };
  assert.equal(
    diffGraph({ ...detail, changes: [invalidated] }, "after").edges[0]
      .diff_status,
    "invalidated",
  );
  assert.equal(
    diffGraph({ ...detail, changes: [invalidated] }, "before").edges[0]
      .diff_status,
    "before",
  );
  assert.equal(
    recordInPhase(
      [
        {
          ...patch("inactive"),
          fields: [
            {
              path: ["active"],
              before: { present: true, value: false },
              after: { present: true, value: false },
            },
          ],
        },
      ],
      "after",
    ),
    false,
  );
});
test("absence and null remain distinguishable", () => {
  assert.equal(fieldText({ present: false, value: null }), "(absent)");
  assert.equal(fieldText({ present: true, value: null }), "null");
});
test("retired nodes are ghosted and invalidated fields have a distinct status", () => {
  const retired = { ...patch("target"), kind: "entity", action: "retire" };
  const graph = diffGraph({ ...detail, changes: [...detail.changes, retired] });
  assert.equal(graph.nodes.find((node) => node.id === "target")?.ghost, true);
  assert.equal(
    changeStatus({
      ...patch("edge-a"),
      fields: [
        {
          path: ["invalid_at"],
          before: { present: false, value: null },
          after: { present: true, value: "2026-10-01T00:00:00Z" },
        },
      ],
    }),
    "invalidated",
  );
});

test("graph slices retain historical knowledge of endpoint changes on other pages", () => {
  const renamed = {
    ...patch("source"),
    kind: "entity",
    fields: [
      {
        path: ["properties", "name"],
        before: { present: true, value: "Historical endpoint" },
        after: { present: true, value: "Renamed endpoint" },
      },
    ],
  };
  const receipt = { ...detail, changes: [...detail.changes, renamed] };
  const graph = diffGraph(receipt, "before", new Set(["edge-a"]));
  assert.equal(graph.edges.length, 1);
  assert.equal(
    graph.nodes.find((node) => node.id === "source")?.caption,
    "Historical endpoint",
  );
  assert.equal(graph.nodes.length, 2);
});
