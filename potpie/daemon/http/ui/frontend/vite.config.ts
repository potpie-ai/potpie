import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The daemon's discovery file: its base URL and the bearer token that /rpc and
// /ui/api both take. Read here so `npm run dev` needs no setup beyond a running
// daemon — the token is rewritten on every `potpie daemon restart`, so copying
// it into a .env would go stale silently.
function daemonDiscovery(): { base_url?: string; token?: string } {
  const home = process.env.CONTEXT_ENGINE_HOME || join(homedir(), ".potpie");
  try {
    return JSON.parse(readFileSync(join(home, "discovery.json"), "utf8"));
  } catch {
    return {};
  }
}

const discovery = daemonDiscovery();
const target =
  process.env.POTPIE_DAEMON_URL || discovery.base_url || "http://127.0.0.1:8099";

// Served by the daemon under /ui, so assets must resolve relative to /ui/.
// For local dev: start the daemon (`potpie setup` / `potpie daemon restart`),
// run `npm run dev`, then browse http://localhost:5173/ui/.
export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/ui/api": {
        target,
        changeOrigin: true,
        // /ui/api is authenticated, and the dev server is a different origin
        // from the daemon: the browser has no session cookie for it (`potpie
        // ui` mints one HttpOnly + SameSite=strict for the daemon's own
        // origin), so every read 401s, and `pots/use` additionally 403s on the
        // same-origin check with `Origin: http://localhost:5173`. The proxy
        // stands in for the CLI — it is the party that can read the token off
        // disk — and speaks as the daemon's own origin. Lowercase keys on
        // purpose: Node lowercases the incoming header names, so these replace
        // them instead of being sent alongside.
        headers: {
          ...(discovery.token
            ? { authorization: `Bearer ${discovery.token}` }
            : {}),
          origin: target,
        },
      },
    },
  },
});
