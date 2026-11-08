// OFM unit testing configuration file.

import { defineConfig } from "vitest/config";
import vue from "@vitejs/plugin-vue";
import path from "path";

export default defineConfig({
  plugins: [
    vue({
      template: {
        transformAssetUrls: { includeAbsolute: false },
        compilerOptions: {
          // This configuration is set to avoid vue warns about missing refs.
          // Forces the test to render all html as dynamic, and avoid hoisting.
          hoistStatic: false,
        },
      },
    }),
  ],

  test: {
    // Set localStorage mock
    setupFiles: ["./src/tests/unit/setup.js"],
    // CSS is disabled for now, the test loads the DOM without styles.
    css: false,
    pool: "forks",
    // Runs in a single thread to avoid CI pipeline crashing.
    singleThread: true,

    // Switch loading vue features
    define: {
      // Feature not needed for current testing
      __VUE_PROD_DEVTOOLS__: false,
      // We use and enforce VUE optionsAPI
      __VUE_OPTIONS_API__: true,
    },

    // Coverage analysis tools setup
    coverage: {
      enabled: true,
      provider: "v8",
      clean: true,
      cleanOnStart: true,
      // text gives output on STOUT and html creates an artifact
      reporter: ["text"],
      all: true,
      // Files to include or exclude in coverage
      include: ["src/**/*.vue", "src/**/*.js"],
      // Exclude from coverage all files with keyword in path and name "Experimental"
      exclude: ["*/experimental/*", "experimental"],
      // These thresholds raise ERROR if the coverage it's under the provided percentages.
      // As the coverage is low we set the limit low, as testing spec files increase,
      // these thresholds need to increase accordingly.
      thresholds: {
        lines: 4,
        statements: 3,
        functions: 2,
        branches: 1,
      },
    },
    // Add largest component dependencies here
    // This chunks together the largest libraries to avoid recurrent parsing and loading
    // and increasing memory overhead. Good for CI.
    deps: {
      optimizer: {
        web: {
          enabled: true,
          include: ["uikit", "openseadragon", "material-symbols"],
        },
      },
    },

    // This is the package responsible for simulating the DOM the other option is JSDOM
    // Happy-dom is the most efficient in memory and its future proof for later node versions.
    environment: "happy-dom",

    globals: true, // Probably good for Mixins

    // This tells vitest to not skip loading these packages,
    // and load them from the local node_modules at run time.
    server: {
      deps: {
        inline: ["vue", "@vue/test-utils", "pinia", "@pinia/testing"],
      },
    },
  },

  // This is to enforce the use of specific packages versions on the testing.
  // This one for example says to load explicitly the vue package at the project's level,
  // and not the global vue if existing.
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      vue: path.resolve(__dirname, "./node_modules/vue"),
    },
  },
});
