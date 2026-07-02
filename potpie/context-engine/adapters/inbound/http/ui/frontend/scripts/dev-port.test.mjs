import assert from "node:assert/strict";
import test from "node:test";

import {
  assertNoCliArgs,
  isAddressInUse,
  portRange,
  startServerInRange,
} from "./dev-port.mjs";

test("assertNoCliArgs accepts the plain dev command", () => {
  assert.doesNotThrow(() => assertNoCliArgs([]));
});

test("assertNoCliArgs rejects ignored Vite CLI flags loudly", () => {
  assert.throws(
    () => assertNoCliArgs(["--host", "0.0.0.0"]),
    (error) => {
      assert.equal(error.code, "POTPIE_UI_DEV_UNSUPPORTED_ARGS");
      assert.match(error.message, /--host 0\.0\.0\.0/);
      return true;
    },
  );
});

test("portRange returns an inclusive ordered TCP range", () => {
  assert.deepEqual(portRange(3000, 3003), [3000, 3001, 3002, 3003]);
});

test("portRange rejects invalid bounds", () => {
  assert.throws(() => portRange(3002, 3001), RangeError);
  assert.throws(() => portRange(0, 3001), RangeError);
  assert.throws(() => portRange(3000.5, 3001), TypeError);
});

test("isAddressInUse recognizes direct and wrapped bind failures", () => {
  const direct = Object.assign(new Error("busy"), { code: "EADDRINUSE" });
  const wrapped = new Error("wrapped", { cause: direct });
  const viteStrictPort = new Error("Port 3000 is already in use");

  assert.equal(isAddressInUse(direct), true);
  assert.equal(isAddressInUse(wrapped), true);
  assert.equal(isAddressInUse(viteStrictPort), true);
  assert.equal(isAddressInUse(new Error("other")), false);
});

test("startServerInRange binds the first available port with strictPort", async () => {
  const calls = [];
  const server = {
    listen: async () => {},
    close: async () => {},
  };

  const result = await startServerInRange({
    minPort: 3000,
    maxPort: 3001,
    createServer: async (config) => {
      calls.push(config);
      return server;
    },
  });

  assert.equal(result.port, 3000);
  assert.equal(result.server, server);
  assert.deepEqual(calls, [
    {
      server: {
        port: 3000,
        strictPort: true,
      },
    },
  ]);
});

test("startServerInRange skips occupied ports and closes failed servers", async () => {
  const closedPorts = [];

  const result = await startServerInRange({
    minPort: 3000,
    maxPort: 3002,
    createServer: async (config) => {
      const port = config.server.port;
      return {
        listen: async () => {
          if (port < 3002) {
            throw Object.assign(new Error(`busy ${port}`), {
              code: "EADDRINUSE",
            });
          }
        },
        close: async () => {
          closedPorts.push(port);
        },
      };
    },
  });

  assert.equal(result.port, 3002);
  assert.deepEqual(closedPorts, [3000, 3001]);
});

test("startServerInRange fails clearly when the range is exhausted", async () => {
  await assert.rejects(
    () =>
      startServerInRange({
        minPort: 3000,
        maxPort: 3001,
        createServer: async () => ({
          listen: async () => {
            throw Object.assign(new Error("busy"), { code: "EADDRINUSE" });
          },
        }),
      }),
    (error) => {
      assert.equal(error.code, "POTPIE_UI_DEV_PORTS_EXHAUSTED");
      assert.match(error.message, /3000-3001/);
      return true;
    },
  );
});

test("startServerInRange rethrows non-port failures", async () => {
  const failure = new Error("config failed");

  await assert.rejects(
    () =>
      startServerInRange({
        minPort: 3000,
        maxPort: 3001,
        createServer: async () => ({
          listen: async () => {
            throw failure;
          },
        }),
      }),
    failure,
  );
});
