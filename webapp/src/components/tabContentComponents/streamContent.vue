<template>
  <div
    id="stream-display"
    ref="streamDisplay"
    class="stream-display uk-width-1-1 uk-height-1-1 scrollTarget"
  >
    <img
      v-show="isVisible && streamEnabled"
      ref="click-frame"
      class="uk-align-center uk-margin-remove-bottom stream-image"
      :style="rotateStyle"
      :src="streamImgUri"
      alt="Stream"
      @load="updateStreamAspectRatio"
      @dblclick="clickMonitor"
    />

    <div v-if="!streamEnabled" class="uk-height-1-1">
      <div v-if="waiting" class="uk-position-center">
        <div uk-spinner="ratio: 4.5"></div>
      </div>

      <div v-else-if="disableStream" class="uk-position-center position-relative text-center">
        Stream preview disabled
      </div>

      <div v-else class="uk-position-center position-relative text-center">
        No active connection
      </div>
    </div>
  </div>
</template>

<script>
import { eventBus } from "../../eventBus.js";
import { useIntersectionObserver } from "@vueuse/core";
import { useSettingsStore } from "@/stores/settings.js";
import { mapState } from "pinia";

const ONE_PIXEL_FALLBACK = "data:image/PNG;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

// Export main app
export default {
  name: "StreamDisplay",

  props: {
    streamId: {
      type: [String],
      required: true,
    },
  },

  setup() {
    const store = useSettingsStore();
    return {
      store,
    };
  },

  data: function () {
    return {
      isVisible: false,
      displaySize: [0, 0],
      displayPosition: [0, 0],
      resizeTimeoutId: null,
    };
  },

  computed: {
    ...mapState(useSettingsStore, [
      "ready",
      "baseUri",
      "disableStream",
      "waiting",
      "navigationRotate",
    ]),
    rotateStyle: function () {
      const rot = (((this.navigationRotate || 0) % 360) + 360) % 360;
      if (rot === 90 || rot === 270) {
        return {
          transform: `rotate(${rot}deg)`,
          maxWidth: "100%",
          maxHeight: "100%",
          width: "auto",
          height: "100%",
        };
      }
      return rot === 180 ? { transform: "rotate(180deg)" } : {};
    },
    streamEnabled: function () {
      return this.ready && !this.disableStream;
    },
    streamImgUri: function () {
      // Only request the real stream if it's enabled AND currently visible on screen
      if (this.isVisible && this.streamEnabled) {
        const url = new URL(`${this.baseUri}/camera/mjpeg_stream`);
        url.searchParams.append("streamId", this.streamId);
        return url.toString();
      }

      // Force the browser to kill the connection by loading a 1 pixel image
      return ONE_PIXEL_FALLBACK;
    },
  },

  mounted() {
    // Initial Timeout
    this.resizeTimeoutId = setTimeout(this.handleDoneResize, 500);

    // set up an intersection observer
    // We capture the 'stop' function returned by VueUse so we can kill it later
    const { stop } = useIntersectionObserver(
      this.$refs.streamDisplay,
      ([{ isIntersecting }]) => {
        this.isVisible = isIntersecting;
      },
      { threshold: 0.0 },
    );
    this.stopIntersectionObserver = stop;
    // A global signal listener to flash the stream element
    eventBus.on("globalFlashStream", this.flashStream);

    // Mutation observer
    this.sizeObserver = new ResizeObserver(() => {
      this.handleResize(); // For any element attached to the observer, run handleResize() on change
    });

    // Safely fetch the DOM element (handles both native HTML and Vue components)
    const streamElement = this.$refs.streamDisplay?.$el || this.$refs.streamDisplay;

    if (streamElement && streamElement.parentNode) {
      // Attach streamDisplay to the size observer
      this.sizeObserver.observe(streamElement.parentNode);
    }
  },

  created: function () {
    // Do nothing: preview stream now runs all the time
  },

  beforeUnmount() {
    // Kill the resize timeout
    clearTimeout(this.resizeTimeoutId);

    // Kill the flash animation timeout (Fixes Beth's MR comment)
    if (this.flashTimeoutId) {
      clearTimeout(this.flashTimeoutId);
    }

    // Kill the Intersection Observer
    if (this.stopIntersectionObserver) {
      this.stopIntersectionObserver();
    }

    // Unregister the event bus listener
    eventBus.off("globalFlashStream", this.flashStream);

    // Disconnect the size Observer
    if (this.sizeObserver) {
      this.sizeObserver.disconnect();
      this.store.removeStream(this.streamId);
    }
    const imgElement = this.$refs["click-frame"];
    if (imgElement) {
      imgElement.src = ONE_PIXEL_FALLBACK;
    }
  },

  methods: {
    flashStream: function () {
      // Run an animation that flashes the stream (for capture feedback)
      let element = this.$refs.streamDisplay;

      // Defensive check in case the ref isn't available
      if (!element) return;

      element.classList.remove("uk-animation-fade");
      /* trigger reflow */
      element.offsetHeight;
      element.classList.add("uk-animation-fade");

      // Clear the timer if flashStream is called again before the 800ms is up
      if (this.flashTimeoutId) {
        clearTimeout(this.flashTimeoutId);
      }

      // Track the timeout ID
      this.flashTimeoutId = setTimeout(() => {
        // Extra defensive check just in case
        if (element && element.classList) {
          element.classList.remove("uk-animation-fade");
        }
      }, 800);
    },

    clickMonitor: function (event) {
      // Calculate steps from event coordinates
      let xCoordinate = event.offsetX;
      let yCoordinate = event.offsetY;

      // Simply scaling by naturalHeight/offsetHeight may give the wrong answer!
      // because we use content-fit: contain in the stylesheet, the img element
      // may be larger than the picture.  So, we must determine whether the
      // width or height is setting the scaling factor, and use a uniform scale
      // factor.
      let scale = Math.max(
        event.target.naturalWidth / event.target.offsetWidth,
        event.target.naturalHeight / event.target.offsetHeight,
      );

      let xRelative = (0.5 * event.target.offsetWidth - xCoordinate) * scale;
      let yRelative = (0.5 * event.target.offsetHeight - yCoordinate) * scale;

      // NB offsetX/offsetY are element-local: the browser already maps the
      // click through the inverse of the preview CSS rotation, so these are
      // raw camera-frame coordinates even when the preview is rotated —
      // no extra transform is wanted here.

      // Emit a signal to move, acted on by paneControl.vue
      eventBus.emit("globalMoveInImageCoordinatesEvent", {
        x: -xRelative,
        y: -yRelative,
      });
    },

    handleResize: function () {
      // Only fires resize event after no resize in 500ms (prevents resize event spam)
      clearTimeout(this.resizeTimeoutId);
      this.resizeTimeoutId = setTimeout(this.handleDoneResize, 250);
    },

    handleDoneResize: function () {
      // Recalculate size
      this.recalculateSize();
      // Handle closed stream
      if (this.displaySize[0] == 0 && this.displaySize[1] == 0) {
        // If this stream was previously active
        if (this.store.activeStreams[this.streamId] == true) {
          this.store.removeStream(this.streamId);
        }
        // If resized to anything other than zero
      } else {
        this.store.addStream(this.streamId);
      }
    },

    recalculateSize: function () {
      // Calculate stream size
      let element = this.$refs.streamDisplay.parentNode;
      let bound = element.getBoundingClientRect();

      let elementSize = [bound.width, bound.height];

      let elementPositionOnWindow = [bound.left, bound.top];
      let windowPositionOnDisplay = [window.screenX, window.screenY];
      let windowChromeHeight = window.outerHeight - window.innerHeight;
      let elementPositionOnDisplay = [
        Math.max(0, windowPositionOnDisplay[0] + elementPositionOnWindow[0]),
        Math.max(0, windowPositionOnDisplay[1] + elementPositionOnWindow[1] + windowChromeHeight),
      ];

      this.displaySize = elementSize;
      this.displayPosition = elementPositionOnDisplay;
    },
    updateStreamAspectRatio(event) {
      if (!this.store.cameraStreamAspectRatio) {
        const img = event.target;

        if (img && img.naturalWidth > 1 && img.naturalHeight > 1) {
          this.store.setCameraStreamAspectRatio(img.naturalWidth, img.naturalHeight);
        }
      }
    },
  },
};
</script>

<style scoped lang="less">
.stream-display img {
  height: 100%;
  text-align: center;
  object-fit: contain;
}

.stream-display {
  width: 100%;
  height: 100%;
}

.position-relative {
  position: relative !important;
}

.text-center {
  text-align: center;
}
</style>
