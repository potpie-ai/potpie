import test from "node:test";
import assert from "node:assert/strict";
import {
  initialTarget,
  isMixed,
  parsePotKey,
  potKey,
  potLabel,
  sameTarget,
} from "../src/hosts.ts";
import type { PotRef, PotsResponse } from "../src/types.ts";

const pots: PotRef[] = [
  { id: "pot_1", name: "default", origin: "local", active: true },
  { id: "pot_1", name: "shop", origin: "managed" },
  { id: "pot_2", name: "default", origin: "managed", active: true },
];

const listing: Pick<PotsResponse, "pots" | "active" | "active_origin"> = {
  pots,
  active: { id: "pot_1", name: "default", origin: "local" },
  active_origin: "local",
};

test("a pot key carries its host, and only known hosts parse", () => {
  assert.equal(potKey({ id: "pot_1", origin: "managed" }), "managed:pot_1");
  assert.deepEqual(parsePotKey("managed:pot:1"), { origin: "managed", id: "pot:1" });
  assert.equal(parsePotKey("elsewhere:pot_1"), null);
  assert.equal(parsePotKey("pot_1"), null);
});

test("the same id on two hosts is two pots", () => {
  assert.equal(
    sameTarget({ id: "pot_1", origin: "local" }, { id: "pot_1", origin: "managed" }),
    false,
  );
});

test("a requested pot is looked up on the requested host only", () => {
  assert.deepEqual(initialTarget(listing, "pot_1", "managed"), {
    id: "pot_1",
    origin: "managed",
  });
  assert.deepEqual(initialTarget(listing, "default", "managed"), {
    id: "pot_2",
    origin: "managed",
  });
  // Not on that host: fall back to the active pot rather than the other host's.
  assert.deepEqual(initialTarget(listing, "shop", "local"), {
    id: "pot_1",
    origin: "local",
  });
});

test("without a request the active host's active pot opens", () => {
  assert.deepEqual(initialTarget(listing, null, null), {
    id: "pot_1",
    origin: "local",
  });
  assert.deepEqual(
    initialTarget({ ...listing, active: null, active_origin: "managed" }, null, null),
    { id: "pot_2", origin: "managed" },
  );
});

test("labels name the host only when both are listed", () => {
  assert.equal(isMixed(pots), true);
  assert.equal(
    potLabel({ id: "p", name: "api", origin: "managed", counts: { claims: 4 } }, true),
    "api (managed) - 4 claims",
  );
  assert.equal(
    potLabel({ id: "p", name: "api", origin: "local" }, false),
    "api",
  );
});
