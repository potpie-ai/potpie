import test from "node:test";
import assert from "node:assert/strict";
import { mutationInContext, mergeContext } from "../src/mutationContext.ts";
import type { CommitDetail, RecordChange } from "../src/commitTypes.ts";
import type { GraphData, GraphNode } from "../src/types.ts";

const liveNode = (key: string, id: string): GraphNode => ({
  id: key,
  key,
  caption: key,
  type: "Service",
  labels: ["Service"],
  properties: { record_id: id, name: key },
});
const change = (
  id: string,
  key: string,
  kind = "entity",
  action = "patch",
): RecordChange => ({
  record_id: id,
  logical_key: key,
  kind,
  action,
  fields: [],
  before_record: null,
  after_record: null,
});
const detail = (changes: RecordChange[]): CommitDetail =>
  ({
    changes,
    display_context: changes
      .filter((c) => c.kind === "entity")
      .map((c) => ({
        record_id: c.record_id,
        logical_key: c.logical_key,
        labels: ["Service"],
        name: c.logical_key,
        kind: "entity",
        endpoint_record_ids: [],
      })),
  }) as CommitDetail;
const live: GraphData = {
  nodes: [
    liveNode("service:a", "a"),
    liveNode("service:b", "b"),
    liveNode("service:c", "c"),
    liveNode("service:unrelated", "u"),
  ],
  edges: [
    { id: "ab", source: "service:a", target: "service:b", predicate: "USES" },
    { id: "bc", source: "service:b", target: "service:c", predicate: "USES" },
  ],
};

test("matches an updated node to the existing graph and mutes its unchanged neighborhood", () => {
  const graph = mutationInContext(detail([change("a", "service:a")]), live);
  assert.deepEqual(
    graph.nodes.map((n) => n.id),
    ["a", "context:service:b"],
  );
  assert.equal(graph.nodes[0].diff_status, "modified");
  assert.equal(graph.nodes[1].diff_status, "context");
  assert.equal(graph.edges.length, 1);
  assert.equal(graph.edges[0].diff_status, "context");
  assert.equal(graph.edges[0].source, "a");
});

test("replaces only the changed relationship while preserving parallel unchanged claims", () => {
  const receipt = detail([change("ab", "claim:ab", "claim", "create")]);
  receipt.changes[0].after_record = {
    fields: {
      subject_record_id: "a",
      object_record_id: "b",
      predicate: "USES",
    },
  };
  receipt.display_context = detail([
    change("a", "service:a"),
    change("b", "service:b"),
  ]).display_context;
  const graph = mutationInContext(receipt, {
    ...live,
    edges: [
      ...live.edges,
      {
        id: "parallel",
        source: "service:a",
        target: "service:b",
        predicate: "USES",
      },
    ],
  });
  assert.equal(graph.edges.filter((e) => e.record_id === "ab").length, 1);
  assert.equal(graph.edges.filter((e) => e.id === "context:ab").length, 0);
  assert.equal(
    graph.edges.find((e) => e.id === "context:parallel")?.diff_status,
    "context",
  );
  assert.equal(
    graph.edges.find((e) => e.record_id === "ab")?.diff_status,
    "added",
  );
  assert.ok(graph.nodes.every((n) => n.diff_status === "context"));
});

test("does not attach an old incarnation to the connections of a recreated entity", () => {
  const graph = mutationInContext(
    detail([change("old-a", "service:a", "entity", "retire")]),
    live,
  );
  assert.deepEqual(
    graph.nodes.map((n) => n.id),
    ["old-a"],
  );
  assert.equal(graph.edges.length, 0);
  assert.equal(graph.nodes[0].diff_status, "retired");
});

test("legacy nodes can match unique logical keys but ambiguous incarnations stay separate", () => {
  const legacy = {
    ...live,
    nodes: live.nodes.map((n) => ({ ...n, properties: {} })),
  };
  assert.equal(
    mutationInContext(detail([change("a", "service:a")]), legacy).edges.length,
    1,
  );
  const ambiguous = mutationInContext(
    detail([
      change("old-a", "service:a", "entity", "retire"),
      change("new-a", "service:a", "entity", "create"),
    ]),
    legacy,
  );
  assert.equal(ambiguous.edges.length, 0);
  assert.equal(ambiguous.nodes.length, 2);
});

test("expansion adds another local neighborhood and leaves source data unchanged", () => {
  const original = JSON.stringify(live);
  const graph = mutationInContext(detail([change("a", "service:a")]), live, [
    "context:service:b",
  ]);
  assert.ok(graph.nodes.some((n) => n.id === "context:service:c"));
  assert.ok(!graph.nodes.some((n) => n.key === "service:unrelated"));
  assert.equal(graph.edges.length, 2);
  assert.equal(JSON.stringify(live), original);
});

test("context limits never hide changed records in a 1,000-record mutation", () => {
  const changes = Array.from({ length: 1000 }, (_, i) =>
    change(`r${i}`, `service:${i}`),
  );
  const existing: GraphData = {
    nodes: changes.flatMap((c, i) => [
      liveNode(c.logical_key, c.record_id),
      liveNode(`neighbor:${i}`, `n${i}`),
    ]),
    edges: changes.map((c, i) => ({
      id: `edge:${i}`,
      source: c.logical_key,
      target: `neighbor:${i}`,
      predicate: "USES",
    })),
  };
  const graph = mutationInContext(detail(changes), existing);
  assert.equal(
    graph.nodes.filter((n) => n.diff_status === "modified").length,
    1000,
  );
  assert.equal(
    graph.nodes.filter((n) => n.diff_status === "context").length,
    120,
  );
  assert.equal(graph.truncated, true);
  const ids = new Set(graph.nodes.map((n) => n.id));
  assert.ok(
    graph.edges.every(
      (e) => ids.has(e.source as string) && ids.has(e.target as string),
    ),
  );
});

test("merging expanded context deduplicates by identity and preserves truncation", () => {
  const merged = mergeContext(
    { ...live, truncated: true },
    { ...live, nodes: [{ ...live.nodes[0], caption: "Updated" }] },
  );
  assert.equal(merged.nodes.length, 4);
  assert.equal(merged.edges.length, 2);
  assert.equal(merged.nodes[0].caption, "Updated");
  assert.equal(merged.truncated, true);
});
