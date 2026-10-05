import test from "node:test";
import assert from "node:assert/strict";
import {
  compactChanges,
  filterChanges,
  groupRecords,
  recordFields,
  recordStatus,
  recordTitle,
} from "../src/commitPresentation.ts";
import type { RecordChange, RecordedContext } from "../src/commitTypes.ts";

const patch = (id: string, status = "patch"): RecordChange => ({
  record_id: id,
  logical_key: "service:checkout",
  kind: "entity",
  action: status,
  fields: [
    {
      path: ["summary"],
      before: { present: true, value: "Before" },
      after: { present: true, value: "After" },
    },
  ],
  before_record: null,
  after_record: null,
});

test("repeated effects group by incarnation without losing their order", () => {
  const first = patch("one");
  const second = patch("two");
  const third = patch("one", "retire");
  assert.deepEqual(groupRecords([first, second, third]), [
    [first, third],
    [second],
  ]);
});

test("audit stamps do not count a newly added record as modified as well", () => {
  assert.equal(recordStatus([patch("one", "create"), patch("one")]), "added");
  assert.equal(recordStatus([patch("one", "retire"), patch("one")]), "retired");
});

test("only fully loaded small mutations expand their diffs", () => {
  assert.equal(compactChanges([patch("one")], 1, false), true);
  assert.equal(compactChanges([patch("one")], 1000, true), false);
  assert.equal(compactChanges([patch("one")], 1, true), false);
  const large = {
    ...patch("one"),
    fields: [],
    after_record: {
      fields: Object.fromEntries(
        Array.from({ length: 13 }, (_, i) => [`field${i}`, i]),
      ),
    },
  };
  assert.equal(compactChanges([large], 1, false), false);
});

test("snapshot comparisons preserve missing versus null and removed fields", () => {
  const change = {
    ...patch("one"),
    fields: [],
    before_record: { fields: { removed: "old", nullable: null } },
    after_record: { fields: { nullable: null, added: "new" } },
  };
  const fields = recordFields(change);
  assert.deepEqual(fields.find((field) => field.path[0] === "removed")?.after, {
    present: false,
    value: undefined,
  });
  assert.deepEqual(
    fields.find((field) => field.path[0] === "nullable")?.after,
    { present: true, value: null },
  );
  assert.deepEqual(fields.find((field) => field.path[0] === "added")?.before, {
    present: false,
    value: undefined,
  });
});

test("large nested snapshots stay collapsed and their fields remain searchable", () => {
  const change = {
    ...patch("one", "create"),
    fields: [],
    after_record: {
      fields: {
        properties: Object.fromEntries(
          Array.from({ length: 30 }, (_, i) => [`field${i}`, i]),
        ),
      },
    },
  };
  assert.equal(recordFields(change).length, 30);
  assert.equal(compactChanges([change], 1, false), false);
  assert.equal(
    filterChanges([change], new Map(), "properties.field29", "added").length,
    1,
  );
});

test("record filters combine status with names, fields and incarnation IDs", () => {
  const context = new Map<string, RecordedContext>([
    [
      "one",
      {
        record_id: "one",
        logical_key: "service:checkout",
        kind: "entity",
        labels: ["Service"],
        name: "Checkout API",
        endpoint_record_ids: [],
      },
    ],
  ]);
  const changes = [patch("one"), patch("two", "retire")];
  assert.deepEqual(
    filterChanges(changes, context, " checkout api ", "modified").map(
      (change) => change.record_id,
    ),
    ["one"],
  );
  assert.deepEqual(
    filterChanges(changes, context, "summary", "retired").map(
      (change) => change.record_id,
    ),
    ["two"],
  );
  assert.deepEqual(
    filterChanges(changes, context, "two", "all").map(
      (change) => change.record_id,
    ),
    ["two"],
  );
  assert.equal(
    filterChanges(changes, context, "checkout api", "added").length,
    0,
  );
  assert.equal(recordTitle(changes[0], context), "Checkout API");
});
