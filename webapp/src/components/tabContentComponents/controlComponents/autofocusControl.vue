<template>
  <div>
    <div v-if="mismatch" class="uk-alert-warning uk-margin-small-bottom" uk-alert>
      <span class="material-symbols-outlined ui-element-error-icon"> warning </span>
      Objective mismatch: the stage calibration looks like
      {{ mismatchSource }}, but "{{ mismatch.objective }}" is selected. Autofocus may be searching
      the wrong range.
      <button class="uk-button uk-button-small uk-margin-small-top" @click="suppressWarning">
        Suppress until objective change
      </button>
    </div>
    <p class="menu-subheading">Autofocus</p>
    <div class="uk-margin">
      <action-button
        thing="autofocus"
        action="fast_autofocus"
        :submit-data="{ dz: 2000 }"
        :submit-label="'Autofocus'"
        :button-primary="true"
        :submit-on-event="'globalFastAutofocusEvent'"
        @task-started="onAutofocus"
        @finished="afterAutofocus"
        @error="modalError"
      />
    </div>
  </div>
</template>
<script>
import ActionButton from "../../labThingsComponents/actionButton.vue";
import { eventBus } from "../../../eventBus.js";

export default {
  name: "AutofocusControl",

  components: {
    ActionButton,
  },

  data: function () {
    return {
      isAutofocusing: false,
      mismatch: null,
      mismatchTimer: null,
    };
  },

  computed: {
    mismatchSource: function () {
      if (!this.mismatch) return "";
      if (this.mismatch.source === "calibrated") {
        return `calibrated for ${this.mismatch.objective}`;
      }
      const s = Number(this.mismatch.live_scale);
      const near = Object.entries({
        "20x": 0.47,
        "40x": 0.25,
        "60x": 0.18,
        "100x": 0.11,
      }).reduce((a, b) => (Math.abs(Math.log(b[1] / s)) < Math.abs(Math.log(a[1] / s)) ? b : a));
      return `~${near[0]} magnification`;
    },
  },

  mounted: function () {
    this.pollMismatch();
    this.mismatchTimer = setInterval(() => this.pollMismatch(), 5000);
  },

  beforeUnmount: function () {
    if (this.mismatchTimer) clearInterval(this.mismatchTimer);
  },

  methods: {
    onAutofocus() {
      this.isAutofocusing = true;
    },
    afterAutofocus() {
      this.isAutofocusing = false;
      eventBus.emit("globalUpdatePositionEvent");
    },
    pollMismatch: async function () {
      try {
        const v = await this.readThingProperty("autofocus", "objective_mismatch");
        this.mismatch = v || null;
      } catch {
        this.mismatch = null;
      }
    },
    suppressWarning: async function () {
      const r = await this.invokeAction("autofocus", "suppress_objective_warning", {});
      if (r !== undefined) {
        this.mismatch = null;
      }
    },
  },
};
</script>
