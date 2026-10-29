import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The daemon's bearer token, from the owner-only credential file it rewrites on
// every boot. Read here so `npm run dev` needs no setup beyond a running
// daemon -- a copy in a .env would go stale on the next `potpie daemon restart`.
function daemonToken(): string | undefined {
  const home = process.env.CONTEXT_ENGINE_HOME || join(homedir(), ".potpie");
  try {
    return readFileSync(join(home, "daemon.credential"), "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}

const target = process.env.POTPIE_DAEMON_URL || "http://127.0.0.1:8099";
const token = daemonToken();

// Served by the daemon under /ui, so assets must resolve relative to /ui/.
// For local dev (`npm run dev`) set POTPIE_DAEMON_URL to the explorer URL from
// `potpie daemon status`, then browse http://localhost:5173/ui/.
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
        // from the daemon: this browser has no session cookie for it, and the
        // daemon refuses `Origin: http://localhost:5173`. The proxy stands in
        // for `potpie ui` -- the party that can read the token off disk -- and
        // speaks as the daemon's own origin. Lowercase keys on purpose: they
        // replace the incoming headers instead of being sent alongside them.
        headers: {
          ...(token ? { authorization: `Bearer ${token}` } : {}),
          origin: target,
        },
      },
    },
  },
});
