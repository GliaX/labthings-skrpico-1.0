<!-- Wizard step: calibrate the illumination flash in every streaming mode.

     Runs automatically when the step is shown, so the microscope has
     working illumination before any later calibration step needs it.
     Emits `advance` on success so the wizard continues by itself.
-->
<template>
  <div>
    <h3>Illumination flash calibration</h3>
    <p v-if="state === 'running'">
      Calibrating the LED flash for each streaming mode. The preview will
      switch modes and flash while measuring - this takes about a minute.
    </p>
    <p v-else-if="state === 'done'">
      Flash calibrated for {{ calibratedCount }} streaming mode(s).
    </p>
    <p v-else-if="state === 'error'" class="cal-missing">
      Flash calibration failed: {{ error }}
    </p>
    <p v-if="state === 'error'">
      You can continue without flash calibration, but image brightness will
      not be calibrated.
    </p>
    <div v-if="state === 'error'" class="uk-margin-top">
      <button
        class="uk-button uk-button-primary"
        type="button"
        @click="runCalibration"
      >
        Retry flash calibration
      </button>
      <button
        class="uk-button uk-button-default"
        type="button"
        @click="$emit('advance')"
      >
        Continue without calibrating
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: "StrobeFlashStep",

  emits: ["advance", "awaiting-user"],

  data: function () {
    return {
      state: "running",
      error: null,
      calibratedCount: 0,
    };
  },

  async mounted() {
    this.$emit("awaiting-user", true);
    await this.runCalibration();
  },

  methods: {
    async runCalibration() {
      this.state = "running";
      this.error = null;
      try {
        const invocation = await this.invokeAction(
          "strobe_control",
          "calibrate_all_modes",
          {},
        );
        const url =
          invocation.href ??
          invocation.links?.find((link) => link.rel === "self")?.href;
        if (!url) {
          throw new Error("no invocation URL returned");
        }
        await this.pollUntilComplete(url, null, null, 1000);
        const response = await fetch(`${url}/output`);
        const output = await response.json();
        const results = output.results ?? [];
        this.calibratedCount = results.filter((r) => r.converged).length;
        this.state = "done";
        this.$emit("awaiting-user", false);
        this.$emit("advance");
      } catch (error) {
        this.error = error.message ?? String(error);
        this.state = "error";
        this.$emit("awaiting-user", false);
      }
    },
  },
};
</script>

<style scoped>
.cal-missing {
  color: #f0506e;
  font-weight: 600;
}
</style>
