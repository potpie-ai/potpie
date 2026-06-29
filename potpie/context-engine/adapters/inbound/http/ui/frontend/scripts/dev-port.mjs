export const DEV_PORT_MIN = 3000;
export const DEV_PORT_MAX = 3099;

export function portRange(minPort = DEV_PORT_MIN, maxPort = DEV_PORT_MAX) {
  if (!Number.isInteger(minPort) || !Number.isInteger(maxPort)) {
    throw new TypeError("dev server port bounds must be integers");
  }
  if (minPort < 1 || maxPort > 65535 || minPort > maxPort) {
    throw new RangeError("dev server port bounds must be a valid TCP range");
  }

  return Array.from(
    { length: maxPort - minPort + 1 },
    (_, index) => minPort + index,
  );
}

export function isAddressInUse(error) {
  for (let current = error; current; current = current.cause) {
    if (current && current.code === "EADDRINUSE") {
      return true;
    }
    if (
      current &&
      typeof current.message === "string" &&
      /^Port \d+ is already in use$/.test(current.message)
    ) {
      return true;
    }
  }
  return false;
}

export async function startServerInRange({
  createServer,
  minPort = DEV_PORT_MIN,
  maxPort = DEV_PORT_MAX,
} = {}) {
  if (typeof createServer !== "function") {
    throw new TypeError("createServer is required");
  }

  let lastAddressInUseError;
  for (const port of portRange(minPort, maxPort)) {
    let server;
    try {
      server = await createServer({
        server: {
          port,
          strictPort: true,
        },
      });
      await server.listen();
      return { port, server };
    } catch (error) {
      await closeQuietly(server);
      if (!isAddressInUse(error)) {
        throw error;
      }
      lastAddressInUseError = error;
    }
  }

  const error = new Error(
    `No available port found for the Potpie UI dev server in ${minPort}-${maxPort}.`,
  );
  error.code = "POTPIE_UI_DEV_PORTS_EXHAUSTED";
  error.cause = lastAddressInUseError;
  throw error;
}

async function closeQuietly(server) {
  if (!server || typeof server.close !== "function") {
    return;
  }
  try {
    await server.close();
  } catch {
    // The server may never have finished binding. The original bind error is
    // the useful signal, so cleanup failures are intentionally ignored.
  }
}
