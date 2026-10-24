import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by the FastAPI backend (legumephc.server).
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/legumephc/server/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 5000,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
