#!/usr/bin/env node

import { createServer } from "vite";

import {
  DEV_PORT_MAX,
  DEV_PORT_MIN,
  startServerInRange,
} from "./dev-port.mjs";

try {
  const { port, server } = await startServerInRange({ createServer });
  console.log(
    `Potpie UI dev server selected port ${port} (range ${DEV_PORT_MIN}-${DEV_PORT_MAX}).`,
  );
  server.printUrls();
  server.bindCLIShortcuts({ print: true });
} catch (error) {
  console.error(formatDevServerError(error));
  process.exitCode = 1;
}

function formatDevServerError(error) {
  if (error && error.code === "POTPIE_UI_DEV_PORTS_EXHAUSTED") {
    return [
      error.message,
      "Free one of those ports, or change DEV_PORT_MIN/DEV_PORT_MAX in scripts/dev-port.mjs.",
    ].join("\n");
  }
  return error && error.stack ? error.stack : String(error);
}
