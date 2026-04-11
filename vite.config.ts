import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      // Order matters: more specific first
      { find: "@/components/ui", replacement: path.resolve(__dirname, "./src/components/ui") },
      { find: "@/components", replacement: path.resolve(__dirname, "./app") },
      { find: "@/lib", replacement: path.resolve(__dirname, "./src/lib") },
      { find: "@", replacement: path.resolve(__dirname, "./app") },
    ],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  }
});
