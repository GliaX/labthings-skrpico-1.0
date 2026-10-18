<template>
  <div
    id="stream-display"
    ref="streamDisplay"
    class="stream-display"
    :style="{ aspectRatio: aspectRatio }"
  >
    <img
      v-if="streamImgUri"
      ref="click-frame"
      class="uk-align-center uk-margin-remove-bottom stream-image"
      :style="rotateStyle"
      :src="streamImgUri"
      alt="Stream"
    />
  </div>
</template>

<script>
import { useIntersectionObserver } from "@vueuse/core";
import { useSettingsStore } from "@/stores/settings.js";
import { mapState } from "pinia";

const ONE_PIXEL_FALLBACK = "data:image/PNG;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

// Export main app
export default {
  name: "MiniStreamDisplay",

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
      observerTimeout: null,
    };
  },

  computed: {
    ...mapState(useSettingsStore, [
      "baseUri",
      "disableStream",
      "ready",
      "cameraStreamAspectRatio",
      "navigationRotate",
    ]),
    aspectRatio() {
      const rot = (((this.navigationRotate || 0) % 180) + 180) % 180;
      const base = this.cameraStreamAspectRatio || 4 / 3;
      return rot === 90 ? 1 / base : base;
    },
    rotateStyle() {
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
      // Require BOTH visibility AND the enabled state
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
    const { stop } = useIntersectionObserver(
      this.$refs.streamDisplay,
      ([{ isIntersecting }]) => {
        // Always update local visibility instantly
        this.isVisible = isIntersecting;

        // Clear any previous pending timers to prevent ghost fires
        clearTimeout(this.observerTimeout);

        if (isIntersecting) {
          // If it's TRUE, we don't want to wait! Turn the stream on immediately.
          this.store.addStream(this.streamId);
        } else {
          // If it's FALSE, wait 50ms before telling the store.
          // If a TRUE event fires in the next 50ms (like during a tab switch),
          // this timeout will be cancelled and the stream will never flicker off!
          this.observerTimeout = setTimeout(() => {
            this.store.removeStream(this.streamId);
          }, 50);
        }
      },
      { threshold: 0.0 },
    );
    this.stopIntersectionObserver = stop;
  },

  beforeUnmount() {
    clearTimeout(this.observerTimeout); // Clean up the timer!
    if (this.stopIntersectionObserver) {
      this.stopIntersectionObserver();
    }
    this.store.removeStream(this.streamId);
    // Override img source to single pixel and drop MJPEG stream
    const imgElement = this.$refs["click-frame"];
    if (imgElement) {
      imgElement.src = ONE_PIXEL_FALLBACK;
    }
  },
};
</script>

<style scoped lang="less">
.stream-display img {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: contain;
  border: 1px solid #555;
}

.stream-display {
  width: 100%;
  height: 100%;
  min-height: 0;
  position: relative;
  overflow: hidden;
}

.position-relative {
  position: relative !important;
}

.text-center {
  text-align: center;
}
</style>
