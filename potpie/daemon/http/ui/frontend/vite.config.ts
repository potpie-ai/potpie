import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served by the daemon under /ui, so assets must resolve relative to /ui/.
// For local dev (`npm run dev`) point the proxy at your running daemon port
// (see `cat ~/.potpie/daemon.json`). The dev launcher prefers
// http://localhost:3000/ui/ and falls back through port 3099.
export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/ui/api": {
        target: process.env.POTPIE_DAEMON_URL || "http://127.0.0.1:8099",
        changeOrigin: true,
      },
    },
  },
});
