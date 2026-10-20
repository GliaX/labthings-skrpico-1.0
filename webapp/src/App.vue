<template>
  <div id="app" class="uk-height-1-1 uk-margin-remove uk-padding-remove" :class="handleTheme">
    <!-- this stops the app loading until setConnected is committed in the store, this means
     other components will not load until we have Thing Descriptions. -->
    <loadingContent v-if="!$store.getters.ready" />
    <appContent v-if="$store.getters.ready" />
    <!-- Runtime modals -->
    <div id="modal-center" ref="keyboardManualModal" class="uk-flex-top" uk-modal>
      <div class="uk-modal-dialog uk-modal-body uk-margin-auto-vertical">
        <button class="uk-modal-close-default" type="button" uk-close></button>
        <div
          v-for="shortcut in keyboardManual"
          :key="shortcut.shortcut"
          class="uk-margin-small"
          uk-grid
        >
          <div class="uk-width-small">{{ shortcut.shortcut }}</div>
          <div class="uk-width-expand">{{ shortcut.description }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
// Import components
import appContent from "./components/appContent.vue";
import loadingContent from "./components/loadingContent.vue";
import Mousetrap from "mousetrap";
import { eventBus } from "./eventBus.js";

const move_keys = ["up", "down", "left", "right", "pageup", "pagedown"];

Mousetrap.prototype.stopCallback = function (e, element) {
  // if the element has the class "mousetrap" then no need to stop
  if ((" " + element.className + " ").indexOf(" Mousetrap ") > -1) {
    return false;
  }

  // if we're in a lightbox, stop mousetrap
  if (element.classList.contains("lightbox-link")) {
    return true;
  }

  // stop for input, select, and textarea
  return (
    element.tagName == "INPUT" ||
    element.tagName == "SELECT" ||
    element.tagName == "TEXTAREA" ||
    (element.contentEditable && element.contentEditable == "true")
  );
};

// Export main app
export default {
  name: "App",

  components: {
    appContent,
    loadingContent,
  },

  data: function () {
    return {
      appAvailable: false,
      arrowKeysDown: {},
      keyboardManual: [],
      systemDark: undefined,
      themeObserver: undefined,
      keysDown: new Set(),
      lastJogTime: 0,
      jogDistance: 600,
      jogTime: 300,
    };
  },

  computed: {
    isSystemDark: function () {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return true;
      } else {
        return false;
      }
    },
    handleTheme: function () {
      var isDark = false;
      if (this.$store.state.appTheme == "dark") {
        isDark = true;
      } else if (this.$store.state.appTheme == "system") {
        if (this.systemDark) {
          isDark = true;
        }
      }
      return {
        "uk-light": isDark,
        "uk-background-secondary": isDark,
      };
    },
  },

  mounted() {
    // Query CSS dark theme preference
    var mql = window.matchMedia("(prefers-color-scheme: dark)");
    // Check for system dark theme when mounted
    if (mql.matches) {
      this.systemDark = true;
    }
    // Create a theme observer to watch for changes
    this.themeObserver = mql.addListener((e) => {
      if (e.matches) {
        this.systemDark = true;
      } else {
        this.systemDark = false;
      }
    });
    // Check connection to API
    this.checkConnection();
  },

  created: function () {
    window.addEventListener("beforeunload", this.handleExit);
    // Scrollwheel listener
    window.addEventListener("wheel", this.wheelMonitor);
    // Watch for origin changes
    this.unwatchOriginFunction = this.$store.watch(
      (state, getters) => {
        return getters.baseUri;
      },
      () => {
        this.checkConnection();
      },
    );

    // Keyboard shortcuts
    Mousetrap.bind("?", () => {
      this.toggleModalElement(this.$refs["keyboardManualModal"]); // Calls the mixin
    });

    Mousetrap.bind(
      move_keys,
      (event, key) => {
        event.preventDefault();
        this.keysDown.add(key);
        this.updateJogFromKeys();
      },
      "keydown",
    );

    Mousetrap.bind(
      move_keys,
      (event, key) => {
        event.preventDefault();
        this.keysDown.delete(key);
        this.updateJogFromKeys();
      },
      "keyup",
    );

    this.keyboardManual.push({
      shortcut: "←↑→↓",
      description: "Move the microscope stage",
    });

    this.keyboardManual.push({
      shortcut: "pgup / pgdn",
      description: "Move the microscope focus",
    });

    // Capture
    Mousetrap.bind("c", () => {
      eventBus.emit("globalCaptureEvent", {});
    });
    this.keyboardManual.push({
      shortcut: "c",
      description: "Take a capture",
    });

    // Autofocus
    Mousetrap.bind("a", () => {
      eventBus.emit("globalFastAutofocusEvent", {});
    });
    this.keyboardManual.push({
      shortcut: "a",
      description: "Fast autofocus",
    });

    // Increment/decrement tab
    Mousetrap.bind("shift+down", () => {
      eventBus.emit("globalIncrementTab", {});
    });
    Mousetrap.bind("shift+up", () => {
      eventBus.emit("globalDecrementTab", {});
    });
    this.keyboardManual.push({
      shortcut: "shift+↑ / shift+↓",
      description: "Switch tab",
    });
  },

  beforeUnmount: function () {
    // Disconnect the theme observer
    if (this.themeObserver) {
      this.themeObserver.disconnect();
    }
    // Remove scrollwheel listener
    window.removeEventListener("wheel", this.wheelMonitor);
    // Remove origin watcher
    this.unwatchOriginFunction();
    // Remove key listeners
    Mousetrap.reset();
  },

  methods: {
    async checkConnection() {
      var baseUri = this.$store.getters.baseUri;
      this.$store.commit("changeWaiting", true);
      // TODO: more robust check - e.g. use a microscope Thing
      // TODO: should we purge existing consumedThings?
      try {
        await this.$store.dispatch("wot/fetchThingDescriptions", `${baseUri}/thing_descriptions/`);
        for (let requiredThing of ["camera", "system"]) {
          if (!this.thingAvailable(requiredThing)) {
            throw new Error(`No ${requiredThing} found, the GUI won't work without one.`);
          }
        }
        try {
          let hostname = await this.readThingProperty("system", "hostname");
          this.$store.commit("changeMicroscopeHostname", hostname);
          document.title = `OpenFlexure Microscope: ${hostname}`;
        } catch {
          this.$store.commit("changeMicroscopeHostname", null);
        }
        // start rendering components
        this.$store.commit("setConnected");
        this.$store.commit("setErrorMessage", null);
      } catch (error) {
        this.$store.commit("setErrorMessage", error);
      } finally {
        this.$store.commit("changeWaiting", false);
      }
    },
    handleExit: function () {},

    /**
     *  Handle global mouse wheel events to be associated with navigation
     */
    wheelMonitor: function (event) {
      // Only capture scroll if the event target's parent contains the "scrollTarget" class
      if (
        event.target.parentNode.classList.contains("scrollTarget") ||
        event.target.classList.contains("scrollTarget")
      ) {
        const z_rel = event.deltaY / 100;
        // Emit a signal to move, acted on by panelControl.vue
        const navigationStepSize = this.$store.state.navigationStepSize;
        const z = z_rel * navigationStepSize.z;
        // Don't use `jog() due to variable size of jogs here and the rate limiting in
        // `jog()`. No need to invert on z, as navigationInvert.z isn't exposed.
        this.invokeAction("stage", "jog", { x: 0, y: 0, z: z });
        eventBus.emit("globalUpdatePositionEvent");
      }
    },

    /**
     * Jog for key-presses.
     *
     * This is a similar to the function in stageControlButtons.vue however it uses
     * uses the key repeat to fire in case a key up is missed. It debounces any
     * request to jog that is too recent after the last jog.
     */
    jog(x, y, z) {
      // Manually debounce extra requests from keyboard repeat rate.
      // This is used rather than an interval in case of missing a repeat.
      const now = Date.now();
      const navigationInvert = this.$store.state.navigationInvert;
      if (now - this.lastJogTime < this.jogTime) {
        return;
      }
      this.lastJogTime = now;

      this.invokeAction("stage", "jog", {
        x: x * this.jogDistance * (navigationInvert.x ? -1 : 1),
        y: y * this.jogDistance * (navigationInvert.y ? -1 : 1),
        z: z * this.jogDistance,
      });
      eventBus.emit("globalUpdatePositionEvent");
    },

    /**
     * Stop jogging on key-up
     *
     * This is also similar to the function in stageControlButtons.vue. It handles
     * stopping jogging and resetting the `lastJogTime` so there is no delay when
     * starting a new jog after an old jog finished.
     */
    jogStop() {
      this.invokeAction("stage", "jog", { stop: true });
      this.lastJogTime = 0;
      setTimeout(() => {
        eventBus.emit("globalUpdatePositionEvent");
      }, 100);
    },

    /**
     * Track which keys are still down on keypress (or key repeat).
     */
    updateJogFromKeys() {
      let x = 0,
        y = 0,
        z = 0;
      if (this.keysDown.has("left")) x -= 1;
      if (this.keysDown.has("right")) x += 1;
      if (this.keysDown.has("up")) y += 1;
      if (this.keysDown.has("down")) y -= 1;
      if (this.keysDown.has("pageup")) z += 1;
      if (this.keysDown.has("pagedown")) z -= 1;

      if (x || y || z) {
        this.jog(x, y, z);
      } else {
        this.jogStop();
      }
    },
  },
};
</script>

<style lang="less">
// Basic UIkit CSS
@import "../node_modules/uikit/src/less/uikit.less";
// Custom UIkit CSS modifications
@import "./assets/less/theme.less";

// We override the custom-electron-titlebar z-index
// UIKit lightbox must be able to draw over the titlebar
// as it currently always spawns at the root of the DOM
.titlebar,
.titlebar > * {
  z-index: 1000 !important;
}

#app {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: left;
  height: 100%;
}

body,
html {
  height: 100%;
  overflow: hidden;
}

.uk-disabled {
  pointer-events: none;
  opacity: 0.4;
}

.control-component {
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-gutter: stable;
  width: 300px;
  height: 100%;
  padding: 0;
  background-color: rgba(180, 180, 180, 0.03);
  border-width: 0 1px 0 0;
  border-style: solid;
  border-color: rgba(180, 180, 180, 0.25);
}

.view-component {
  overflow-y: auto;
  overflow-x: hidden;
  height: 100%;
  padding: 0;
}

.image-fit {
  height: 80%;
  width: 100%;
  object-fit: contain;
  overflow-y: clip;
}

.section-content {
  padding: 0;
  height: 100%;
}

.thumbnail-fit {
  max-height: 120px;
  max-width: 240px;
  object-fit: contain;
  overflow-y: hidden;
}
</style>
