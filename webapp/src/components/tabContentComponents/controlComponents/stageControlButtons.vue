<template>
  <div class="uk-flex uk-flex-center uk-flex-middle uk-margin">
    <div
      class="dpad-grid"
      :class="{
        'dpad-only': showDpad && !showFocusControls,
        'focus-only': !showDpad && showFocusControls,
        'both-controls': showDpad && showFocusControls,
      }"
    >
      <button
        v-if="showDpad"
        id="up-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, 0, 1, 0)"
        @pointerup="jogStop()"
        @mouseLeave="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> arrow_upward </span>
      </button>

      <button
        v-if="showDpad"
        id="left-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, -1, 0, 0)"
        @pointerup="jogStop()"
        @pointercancel="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> arrow_back </span>
      </button>

      <button
        v-if="showDpad"
        id="right-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, 1, 0, 0)"
        @pointerup="jogStop()"
        @pointercancel="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> arrow_forward </span>
      </button>

      <button
        v-if="showDpad"
        id="down-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, 0, -1, 0)"
        @pointerup="jogStop()"
        @pointercancel="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> arrow_downward </span>
      </button>

      <button
        v-if="showFocusControls"
        id="focus-out-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, 0, 0, -1)"
        @pointerup="jogStop()"
        @pointercancel="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> remove </span>
      </button>

      <button
        v-if="showFocusControls"
        id="focus-in-button"
        class="uk-button uk-button-primary dpad-btn"
        @pointerdown="jog($event, 0, 0, 1)"
        @pointerup="jogStop()"
        @pointercancel="jogStop()"
      >
        <span class="material-symbols-outlined sync-icon"> add </span>
      </button>
    </div>
  </div>
</template>

<script>
import { eventBus } from "@/eventBus.js";

export default {
  name: "StageControlButtons",
  props: {
    showDpad: {
      type: Boolean,
      default: true,
    },
    showFocusControls: {
      type: Boolean,
      default: true,
    },
  },
  data: () => ({
    jogIntervalId: null,
    jogDistance: 10,
    jogTime: 3,
  }),
  methods: {
    /**
     * Jog d-pad and focus buttons.
     *
     * This is a similar to the function in App.vue, however it uses an Interval rather
     * than the one in App.vue that uses key repeats.
     */
    jog(pointerEvent, x, y, z) {
      // Only respond to primary button (left mouse / primary touch)
      if (pointerEvent.button !== 0) return;

      if (this.jogIntervalId) {
        clearInterval(this.jogIntervalId);
      }

      // Designate this element to get the pointers next pointerup event wherever that
      // pointer is.
      pointerEvent.target.setPointerCapture(pointerEvent.pointerId);

      const navigationInvert = this.$store.state.navigationInvert;
      let invokeJog = () =>
        this.invokeAction("stage", "jog", {
          x: x * this.jogDistance * (navigationInvert.x ? -1 : 1),
          y: y * this.jogDistance * (navigationInvert.y ? -1 : 1),
          z: z * this.jogDistance,
        });
      invokeJog();
      this.jogIntervalId = setInterval(invokeJog, this.jogTime);
    },
    /**
     * Stop jogging from d-pad and focus buttons.
     *
     * This is a similar to the function in App.vue, but it is designed to clear the
     * interval used with the d-pad and focus buttons.
     */
    jogStop() {
      if (this.jogIntervalId) {
        clearInterval(this.jogIntervalId);
      }
      this.invokeAction("stage", "jog", { stop: true });
      setTimeout(() => {
        eventBus.emit("globalUpdatePositionEvent");
      }, 100);
    },
  },
};
</script>

<style scoped>
.dpad-grid {
  display: grid;
  grid-template-columns: repeat(3, 40px);
  gap: 1px;
  justify-content: center;
  align-items: center;
}

.both-controls {
  grid-template-rows: 40px 40px 40px 20px 40px;
}

.dpad-only {
  grid-template-rows: 40px 40px 40px;
}

.focus-only {
  grid-template-rows: 40px;
}

/* Place buttons within grid */
.dpad-grid #up-button {
  grid-column: 2;
  grid-row: 1;
}
.dpad-grid #left-button {
  grid-column: 1;
  grid-row: 2;
}
.dpad-grid #right-button {
  grid-column: 3;
  grid-row: 2;
}
.dpad-grid #down-button {
  grid-column: 2;
  grid-row: 3;
}

.both-controls #focus-out-button {
  grid-column: 1;
  grid-row: 5;
}
.both-controls #focus-in-button {
  grid-column: 3;
  grid-row: 5;
}

.focus-only #focus-out-button {
  grid-column: 1;
  grid-row: 1;
}
.focus-only #focus-in-button {
  grid-column: 3;
  grid-row: 1;
}

.dpad-btn {
  width: 40px;
  height: 40px;
  justify-content: center;
  align-items: center;
  display: flex;
}
</style>
