<template>
  <div id="app" class="uk-height-1-1 uk-margin-remove uk-padding-remove" :class="handleTheme">
    <portraitWarning />

    <!-- this stops the app loading until setConnected is committed in the store,
         this means other components will not load until we have Thing Descriptions. -->
    <loadingContent v-if="!ready" />
    <appContent v-else />
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
import { mapWritableState, mapState, mapActions } from "pinia";
import { useSettingsStore } from "@/stores/settings.js";
import { useWotStore } from "@/stores/wot.js";
import portraitWarning from "./components/genericComponents/portraitWarning.vue";

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
    portraitWarning,
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
      jogTime: 30,
    };
  },

  computed: {
    ...mapWritableState(useSettingsStore, [
      "appTheme",
      "navigationStepSize",
      "navigationInvert",
      "navigationRotate",
      "stageToImageRotation",
      "microscopeHostname",
      "error",
      "waiting",
    ]),
    ...mapState(useSettingsStore, ["ready", "baseUri"]),

    isSystemDark: function () {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return true;
      } else {
        return false;
      }
    },
    handleTheme: function () {
      var isDark = false;
      if (this.appTheme == "dark") {
        isDark = true;
      } else if (this.appTheme == "system") {
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

  // watch origin function refactored
  watch: {
    baseUri() {
      this.checkConnection();
    },
  },

  mounted() {
    // Query CSS dark theme preference
    this.mql = window.matchMedia("(prefers-color-scheme: dark)");
    this.systemDark = this.mql.matches;
    // Check for system dark theme when mounted
    if (this.mql.matches) {
      this.systemDark = true;
    }
    // Create a theme observer to watch for changes
    this.themeWatchdog = (e) => {
      this.systemDark = e.matches;
    };
    this.mql.addEventListener("change", this.themeWatchdog);
    // Check connection to API
    this.checkConnection();
  },

  created: function () {
    window.addEventListener("beforeunload", this.handleExit);
    this.bindHardwareInputs(); // Turn keys ON when connected
  },

  beforeUnmount: function () {
    // Disconnect the theme observer
    this.mql.removeEventListener("change", this.themeWatchdog);
    // Remove scrollwheel listener
    this.unbindHardwareInputs();
  },

  methods: {
    ...mapActions(useSettingsStore, ["setConnected"]),

    bindHardwareInputs() {
      window.addEventListener("wheel", this.wheelMonitor);

      Mousetrap.bind("?", () => {
        this.toggleModalElement(this.$refs["keyboardManualModal"]);
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
      // The following signal is managed (on/off) by actionButton.vue at this.submitOnEvent.
      Mousetrap.bind("c", () => {
        eventBus.emit("globalCaptureEvent", {});
      });
      this.keyboardManual.push({
        shortcut: "c",
        description: "Take a capture",
      });
      // The following signal is managed (on/off) by actionButton.vue at this.submitOnEvent.
      // There's no visual feedback when the event is triggered
      Mousetrap.bind("a", () => {
        eventBus.emit("globalFastAutofocusEvent", {});
      });
      this.keyboardManual.push({
        shortcut: "a",
        description: "Fast autofocus",
      });
      // This signal is managed on lifecycle mounted and unmounted optionsAPI at appContent.vue.
      Mousetrap.bind("shift+down", () => {
        eventBus.emit("globalIncrementTab", {});
      });
      // This signal is managed on lifecycle mounted and unmounted optionsAPI at appContent.vue.
      Mousetrap.bind("shift+up", () => {
        eventBus.emit("globalDecrementTab", {});
      });
      this.keyboardManual.push({
        shortcut: "shift+↑ / shift+↓",
        description: "Switch tab",
      });
    },

    unbindHardwareInputs() {
      window.removeEventListener("wheel", this.wheelMonitor);
      Mousetrap.reset();
      this.keyboardManual = [];
    },

    async checkConnection() {
      this.waiting = true;
      const wotStore = useWotStore();
      try {
        await wotStore.fetchThingDescriptions(`${this.baseUri}/thing_descriptions/`);
        for (let requiredThing of ["camera", "system"]) {
          if (!this.thingAvailable(requiredThing)) {
            throw new Error(`No ${requiredThing} found, the GUI won't work without one.`);
          }
        }
        try {
          let hostname = await this.readThingProperty("system", "hostname");
          this.microscopeHostname = hostname;
          document.title = `OpenFlexure Microscope: ${hostname}`;
        } catch {
          this.microscopeHostname = null;
        }
        this.setConnected();
        this.error = null;
      } catch (error) {
        this.error = error;
      } finally {
        this.waiting = false;
      }
    },

    handleExit() {},

    /**
     *  Handle global mouse wheel events to be associated with navigation
     */
    wheelMonitor(event) {
      if (
        event.target.parentNode.classList.contains("scrollTarget") ||
        event.target.classList.contains("scrollTarget")
      ) {
        const z_rel = event.deltaY / 100;
        const z = z_rel * this.navigationStepSize.z;
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
      const now = Date.now();
      if (now - this.lastJogTime < this.jogTime) {
        return;
      }
      this.lastJogTime = now;
      // Screen intent (x=right, y=down) -> stage axes. The camera shows
      // the stage frame rotated by stageToImageRotation; the preview is
      // then rotated by navigationRotate. To make the view move in the
      // pressed direction, rotate the request by (previewRot - stageRot).
      let jx = x * this.navigationStepSize.x * (this.navigationInvert.x ? -1 : 1);
      let jy = y * this.navigationStepSize.y * (this.navigationInvert.y ? -1 : 1);
      const delta =
        ((((this.navigationRotate || 0) - (this.stageToImageRotation || 0)) % 360) + 360) % 360;
      if (delta === 90) {
        [jx, jy] = [jy, -jx];
      } else if (delta === 180) {
        [jx, jy] = [-jx, -jy];
      } else if (delta === 270) {
        [jx, jy] = [-jy, jx];
      }
      this.invokeAction("stage", "jog", {
        x: jx,
        y: jy,
        z: z * this.navigationStepSize.z,
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
@import url("./assets/less/theme.less");
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
  overflow: hidden auto;
  scrollbar-gutter: stable both-edges;
  width: 300px;
  height: 100%;
  padding: 0;
  background-color: rgba(180, 180, 180, 0.03);
  border-width: 0 1px 0 0;
  border-style: solid;
  border-color: rgba(180, 180, 180, 0.25);
}

.view-component {
  overflow: hidden auto;
  height: 100%;
  padding: 0;
}

.view-component.uk-padding-small {
  padding: 15px;
}

.image-fit-wrapper {
  width: 100%;
  height: 100%;
  min-height: 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 12px;
  box-sizing: border-box;
  overflow: hidden;
}

.image-fit {
  height: 100%;
  width: 100%;
  object-fit: contain;
  overflow-y: clip;
}

.section-content {
  padding: 0;
  height: 100%;
}

.ui-element-error-icon {
  vertical-align: middle;
  padding-right: 10px;
  color: #a00;
}

.ui-element-broken {
  border-color: #a00 !important;
  color: #a00 !important;
  cursor: not-allowed;
}
</style>
