<template>
  <div
    id="stream-display"
    ref="streamDisplay"
    class="stream-display uk-width-1-1 uk-height-1-1 scrollTarget"
  >
    <img
      v-if="isVisible"
      ref="click-frame"
      class="uk-align-center uk-margin-remove-bottom"
      :hidden="!streamEnabled"
      :src="streamImgUri"
      alt="Stream"
      @dblclick="clickMonitor"
    />

    <div v-if="!streamEnabled" class="uk-height-1-1">
      <div v-if="$store.state.waiting" class="uk-position-center">
        <div uk-spinner="ratio: 4.5"></div>
      </div>

      <div
        v-else-if="$store.state.disableStream"
        class="uk-position-center position-relative text-center"
      >
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

// Export main app
export default {
  name: "StreamDisplay",

  data: function () {
    return {
      isVisible: false,
      displaySize: [0, 0],
      displayPosition: [0, 0],
      resizeTimeoutId: setTimeout(this.doneResizing, 500),
    };
  },

  computed: {
    streamEnabled: function () {
      return this.$store.getters.ready && !this.$store.state.disableStream;
    },
    thisStreamOpen: function () {
      // Only a single MJPEG connection should be open at a time
      return !(this.displaySize[0] == 0) && !(this.displaySize[1] == 0);
    },
    streamImgUri: function () {
      return `${this.$store.getters.baseUri}/camera/mjpeg_stream`;
    },
  },

  mounted() {
    //set up an intersection observer
    useIntersectionObserver(
      this.$refs.streamDisplay,
      ([{ isIntersecting }]) => {
        this.isVisible = isIntersecting;
      },
      {
        threshold: 0.0,
      },
    );
    // A global signal listener to flash the stream element
    this.onFlashStream = () => {
      this.flashStream();
    };

    eventBus.on("globalFlashStream", this.onFlashStream);

    // Mutation observer
    this.sizeObserver = new ResizeObserver(() => {
      this.handleResize(); // For any element attached to the observer, run handleResize() on change
    });
    // Fetch streamDisplay component by ref
    if (this.$refs.streamDisplay && this.$refs.streamDisplay.parentNode) {
      const streamDisplayElement = this.$refs.streamDisplay.parentNode;
      // Attach streamDisplay to the size observer
      this.sizeObserver.observe(streamDisplayElement);
    }
  },

  created: function () {
    // Do nothing: preview stream now runs all the time
  },

  beforeUnmount: function () {
    // Disconnect the size observer
    this.sizeObserver.disconnect();
    // Remove from the array of active streams
    this.$store.commit("removeStream", this._uid);
  },

  methods: {
    flashStream: function () {
      // Run an animation that flashes the stream (for capture feedback)
      let element = this.$refs.streamDisplay;
      element.classList.remove("uk-animation-fade");
      element.offsetHeight; /* trigger reflow */
      element.classList.add("uk-animation-fade");
      setTimeout(function () {
        element.classList.remove("uk-animation-fade");
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
        if (this.$store.state.activeStreams[this._uid] == true) {
          this.$store.commit("removeStream", this._uid);
        }
        // If resized to anything other than zero
      } else {
        this.$store.commit("addStream", this._uid);
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
