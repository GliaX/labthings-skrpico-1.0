<template>
  <div class="uk-width-large">
    <p><b>Single Move Step Size</b></p>
    <p>
      This sets the size (and direction) of movements made using the navigation buttons in the
      control tab or using the keyboard (arrow keys, page up/down).
    </p>
    <p>These settings do not affect the operation of other actions your microscope performs.</p>
    <div class="uk-grid-small uk-child-width-1-3" uk-grid>
      <div>
        <label class="uk-form-label" for="form-stacked-text">x</label>
        <div class="uk-form-controls">
          <input v-model="navigationStepSize.x" class="uk-input uk-form-small" type="number" />
        </div>
        <label class="uk-margin-small-right">
          <input v-model="navigationInvert.x" class="uk-checkbox" type="checkbox" />
          Invert x
        </label>
      </div>
      <div>
        <label class="uk-form-label" for="form-stacked-text">y</label>
        <div class="uk-form-controls">
          <input v-model="navigationStepSize.y" class="uk-input uk-form-small" type="number" />
        </div>
        <label class="uk-margin-small-right">
          <input v-model="navigationInvert.y" class="uk-checkbox" type="checkbox" />
          Invert y
        </label>
      </div>
      <div>
        <label class="uk-form-label" for="form-stacked-text">z</label>
        <div class="uk-form-controls">
          <input v-model="navigationStepSize.z" class="uk-input uk-form-small" type="number" />
        </div>
      </div>
    </div>
    <div class="uk-grid-small" uk-grid>
      <div class="uk-width-1-3">
        <label class="uk-form-label" for="form-stacked-select">Preview rotation</label>
        <div class="uk-form-controls">
          <select v-model.number="navigationRotate" class="uk-select uk-form-small">
            <option :value="0">0°</option>
            <option :value="90">90°</option>
            <option :value="180">180°</option>
            <option :value="270">270°</option>
          </select>
        </div>
        <p class="uk-text-meta uk-margin-remove">
          Rotates the camera preview; navigation follows the rotated view. Use Auto-Detect View
          Alignment in the Camera-Stage Mapping settings to measure it automatically.
        </p>
      </div>
    </div>
    <shaper-calibration-card />
  </div>
</template>

<script>
import ShaperCalibrationCard from "./ShaperCalibrationCard.vue";
import { storeToRefs } from "pinia";
import { useSettingsStore } from "@/stores/settings.js";

export default {
  name: "StageControlSettings",

  components: { ShaperCalibrationCard },

  setup() {
    const store = useSettingsStore();
    const { navigationStepSize, navigationInvert, navigationRotate } = storeToRefs(store);
    return { navigationStepSize, navigationInvert, navigationRotate };
  },
};
</script>
