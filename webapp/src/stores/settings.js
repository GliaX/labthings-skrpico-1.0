/** * @module settings
 * @description Settings and state related to the microscope connection.
 */

import { defineStore } from "pinia";
import { ref } from "vue";

/**
 * Get the origin from the current window location.
 * @returns {string} The origin URL (e.g., "http://microscope.local:5000/api/v3").
 */

function getOriginFromLocation() {
  // This will default to the same origin that's serving
  // the web app.
  let url = new URL(window.location.href);
  return `${url.origin}/api/v3`;
}

/**
 * A Pinia store for managing application settings and state related to the microscope connection.
 * This store includes state variables for connection status, error messages, active streams,
 * and user preferences such as theme and navigation settings. It also provides actions to
 * reset the state, set the connection status, and manage active streams.
 */
export const useSettingsStore = defineStore(
  "settings",
  () => {
    // State
    const baseUri = ref(getOriginFromLocation());
    const ready = ref(false);
    const waiting = ref(false);
    const error = ref("");
    const trackWindow = ref(true);
    const activeStreams = ref({});
    const microscopeHostname = ref("");

    // Persistent items:
    // The app theme (e.g. light/dark)
    const appTheme = ref("system");
    const disableStream = ref(false);
    // The origin to use if overriding with dev tools
    const overrideOrigin = ref("http://microscope.local:5000/api/v3");
    // The step sizes for navigation via control pane/keys presses
    const navigationStepSize = ref({
      x: 200,
      y: 200,
      z: 50,
    });
    // The axis inversion for navigation via control pane/keys presses
    const navigationInvert = ref({
      x: false,
      y: false,
      z: false,
    });
    // Preview rotation in degrees (0/90/180/270). Rotates the camera
    // preview and remaps navigation so screen directions match the view.
    const navigationRotate = ref(0);
    // Preview horizontal mirror (optics handedness). Composed with rotation.
    const navigationMirror = ref(false);
    // Measured rotation from stage axes to camera image axes (degrees).
    // Set by the Auto-Detect View Alignment action; 0 before first run.
    const stageToImageRotation = ref(0);
    // The aspect ratio of the camera stream that the webapp started with
    const cameraStreamAspectRatio = ref(null);

    // Actions
    function resetState() {
      waiting.value = false;
      ready.value = false;
      // On resetState there is no connection.
      error.value = "Microscope is not connected.";
    }

    function setConnected() {
      waiting.value = false;
      ready.value = true;
    }

    function addStream(id) {
      activeStreams.value[id] = true;
    }
    function removeStream(id) {
      activeStreams.value[id] = false;
    }

    function setCameraStreamAspectRatio(width, height) {
      cameraStreamAspectRatio.value = width / height;
    }

    // Export
    return {
      // State
      baseUri,
      ready,
      waiting,
      error,
      trackWindow,
      activeStreams,
      microscopeHostname,
      appTheme,
      disableStream,
      overrideOrigin,
      navigationStepSize,
      navigationInvert,
      navigationRotate,
      navigationMirror,
      stageToImageRotation,
      cameraStreamAspectRatio,

      // Actions
      resetState,
      setConnected,
      addStream,
      removeStream,
      setCameraStreamAspectRatio,
    };
  },
  {
    // PiniaPluginPersistedState will now automatically persist ONLY these specific refs to localStorage
    persist: {
      pick: [
        "appTheme",
        "overrideOrigin",
        "disableStream",
        "navigationStepSize",
        "navigationInvert",
        "navigationRotate",
        "navigationMirror",
        "stageToImageRotation",
      ],
    },
  },
);
