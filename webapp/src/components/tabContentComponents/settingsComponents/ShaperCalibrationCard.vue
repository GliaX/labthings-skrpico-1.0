<!-- Input-shaper calibration from the camera (no accelerometer).
     Global-shutter run: strobed frames, clean 20-65 Hz spectrum.
     Rolling-shutter run: constant light, row-wise sampling, sees past 70 Hz.
     Motion is driven by the microscope itself; the stage must be free to move. -->
<template>
  <div class="shaper-card">
    <p><b>Input shaper calibration</b></p>
    <p class="uk-text-meta">
      Rattles the stage through a frequency sweep while the camera watches a
      textured feature, then fits the input shaper that cancels the measured
      vibration. Use a slide with visible structure, focused, with the feature
      near the centre of the preview. The stage moves by itself for ~20 s per
      axis; do not touch the microscope during the run.
    </p>
    <div v-if="!running" class="uk-flex uk-flex-middle" uk-grid>
      <div class="uk-width-auto">
        <span class="uk-form-label uk-margin-right">Axis</span>
        <select v-model="axis" class="uk-select uk-form-width-xsmall">
          <option value="x">x</option>
          <option value="y">y</option>
        </select>
      </div>
      <div class="uk-width-expand uk-text-right">
        <button
          id="shaper-run-strobe"
          class="uk-button uk-button-primary"
          :disabled="running"
          @click="run('strobe')"
        >
          Calibrate (global shutter)
        </button>
        <button
          id="shaper-run-rolling"
          class="uk-button uk-button-default uk-margin-left"
          :disabled="running"
          @click="run('rolling')"
        >
          Calibrate (rolling shutter)
        </button>
      </div>
    </div>
    <p v-if="state === 'running'" class="uk-text-meta">
      Sweeping the {{ axis.toUpperCase() }} axis and recording - about 20 seconds.
      This button disables until the run finishes.
    </p>
    <p v-if="state === 'error'" class="shaper-error">
      Calibration failed: {{ error }}
    </p>
    <div v-if="state === 'done' && result" class="shaper-result">
      <p>
        Measured resonance peak:
        <b>{{ result.dominant_peak_hz.toFixed(1) }} Hz</b>
      </p>
      <p>
        Recommended shaper for
        <b>{{ axis.toUpperCase() }}</b>:
        <b>{{ result.recommended.name }}</b> at
        <b>{{ result.recommended.freq.toFixed(1) }} Hz</b>
        (residual vibrations
        {{ result.recommended.vibrs_pct.toFixed(2) }}%)
      </p>
      <table class="uk-table uk-table-small uk-table-divider shaper-table">
        <thead>
          <tr>
            <th>Shaper</th>
            <th>Frequency</th>
            <th>Residual vib.</th>
            <th>Smoothing</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in result.shaper_table" :key="row.name">
            <td>{{ row.name }}</td>
            <td>{{ row.freq.toFixed(1) }} Hz</td>
            <td>{{ row.vibrs_pct.toFixed(2) }}%</td>
            <td>{{ row.smoothing.toFixed(3) }}</td>
          </tr>
        </tbody>
      </table>
      <p class="uk-text-meta">
        Applied automatically: printer.cfg now carries
        {{ result.applied.shaper_type_x }}/{{ result.applied.shaper_freq_x }} (x)
        and {{ result.applied.shaper_type_y }}/{{ result.applied.shaper_freq_y }} (y),
        and klipper restarted with the new shaper live.
      </p>
      <p v-if="result.applied.warning" class="shaper-warning">
        {{ result.applied.warning }}
      </p>
      <p v-if="result.applied.klipper_restarted" class="uk-text-meta">
        Klipper restarted after the write; restart took a few seconds.
      </p>
    </div>
  </div>
</template>

<script>
export default {
  name: "ShaperCalibrationCard",

  data: function () {
    return {
      axis: "x",
      running: false,
      applied: null,
      state: "idle",
      error: null,
      result: null,
    };
  },

  methods: {
    async run(variant) {
      this.running = true;
      this.state = "running";
      this.error = null;
      this.result = null;
      const action =
        variant === "rolling"
          ? "calibrate_input_shaper_rolling"
          : "calibrate_input_shaper";
      try {
        const invocation = await this.invokeAction("shaper_detect", action, {
          axis: this.axis,
        });
        if (invocation === undefined) {
          throw new Error("invocation failed - see server log");
        }
        const data = invocation.data ?? {};
        const url =
          invocation.headers?.location ??
          data.href ??
          data.links?.find((link) => link.rel === "self")?.href;
        if (!url) {
          throw new Error("no invocation URL in response");
        }
        await this.pollUntilComplete(url, null, null, 1000);
        const response = await fetch(`${url}/output`);
        const output = await response.json();
        this.result = output[this.axis] ?? output;
        this.applied = output.applied ?? null;
        this.state = "done";
      } catch (error) {
        this.error = error.message ?? String(error);
        this.state = "error";
      } finally {
        this.running = false;
      }
    },
  },
};
</script>

<style scoped>
.shaper-card {
  margin-top: 1.5em;
  padding-top: 1em;
  border-top: 1px solid #e5e5e5;
}
.shaper-warning {
  color: #ff7a00;
  font-weight: 600;
}
.shaper-error {
  color: #f0506e;
  font-weight: 600;
}
.shaper-table {
  max-width: 420px;
}
</style>
