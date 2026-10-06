import test from "node:test";
import assert from "node:assert/strict";
import { api } from "../src/api.ts";
import { session, SessionRequiredError } from "../src/session.ts";

type Call = { url: string; init: RequestInit };

function stubFetch(status: number, body: unknown = {}): Call[] {
  const calls: Call[] = [];
  globalThis.fetch = (async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  return calls;
}

test.beforeEach(() => session.reset());

test("every request sends the same-origin session cookie", async () => {
  const calls = stubFetch(200, { pots: [], active: null });

  await api.pots();
  await api.usePot("default");

  assert.deepEqual(
    calls.map((c) => [c.url, c.init.credentials]),
    [
      ["/ui/api/pots", "same-origin"],
      ["/ui/api/pots/use", "same-origin"],
    ],
  );
  assert.equal(calls[1].init.method, "POST");
});

test("a 401 switches the page to the session-required state once", async () => {
  stubFetch(401, { detail: "unauthorized" });
  let notified = 0;
  const unsubscribe = session.subscribe(() => notified++);

  await assert.rejects(api.graph("pot_1"), SessionRequiredError);
  await assert.rejects(api.usePot("default"), SessionRequiredError);
  unsubscribe();

  assert.equal(session.isRequired(), true);
  assert.equal(notified, 1);
});

test("the session-required error tells the user to run potpie ui", async () => {
  stubFetch(401);

  await assert.rejects(api.status(), (error: Error) =>
    error.message.includes("potpie ui"),
  );
});

test("other failures stay ordinary errors with the daemon's detail", async () => {
  stubFetch(409, { detail: "no active pot" });

  await assert.rejects(api.status(), (error: Error) => {
    assert.ok(!(error instanceof SessionRequiredError));
    assert.equal(error.message, "no active pot");
    return true;
  });
  assert.equal(session.isRequired(), false);
});

test("a cross-origin refusal is not mistaken for an expired session", async () => {
  stubFetch(403, { detail: "cross-origin request refused" });

  await assert.rejects(api.usePot("default"), /cross-origin request refused/);
  assert.equal(session.isRequired(), false);
});
