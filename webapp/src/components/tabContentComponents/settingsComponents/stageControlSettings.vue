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
    <div class="uk-margin-top">
      <button class="uk-button uk-button-default" :disabled="detecting" @click="detectRotation">
        {{ detecting ? "Detecting..." : "Detect rotation" }}
      </button>
      <span v-if="detectMessage" class="uk-margin-left">{{ detectMessage }}</span>
    </div>
    <div class="uk-margin-small uk-flex uk-flex-middle">
      <button
        v-for="angle in [0, 90, 180, 270]"
        :key="angle"
        class="uk-button uk-button-small uk-margin-small-right"
        :class="{ 'uk-button-primary': previewRotation === angle }"
        @click="setPreviewTransform(angle, previewMirror)"
      >
        {{ angle }}&deg;
      </button>
      <label class="uk-margin-small-right">
        <input v-model="previewMirror" type="checkbox" class="uk-checkbox" />
        Mirror
      </label>
      <span>Current: {{ previewRotation }}&deg;{{ previewMirror ? ", mirrored" : "" }}</span>
    </div>
    <shaper-calibration-card />
  </div>
</template>

<script>
import ShaperCalibrationCard from "./ShaperCalibrationCard.vue";
import { ref } from "vue";
import { storeToRefs } from "pinia";
import { useSettingsStore } from "@/stores/settings.js";
import { detectRotationFromCSM } from "@/js_utils/previewRotation.mjs";

export default {
  name: "StageControlSettings",

  components: { ShaperCalibrationCard },

  setup() {
    const store = useSettingsStore();
    const { navigationStepSize, navigationInvert, previewRotation, previewMirror } = storeToRefs(store);
    const detecting = ref(false);
    const detectMessage = ref("");
    return {
      navigationStepSize,
      navigationInvert,
      previewRotation,
      previewMirror,
      detecting,
      detectMessage,
      setPreviewTransform: store.setPreviewTransform,
    };
  },

  methods: {
    async detectRotation() {
      this.detecting = true;
      this.detectMessage = "";
      try {
        const matrix = await this.readThingProperty(
          "camera_stage_mapping",
          "image_to_stage_displacement_matrix",
        );
        if (!matrix) {
          this.detectMessage =
            "No calibration available - run the camera-stage mapping calibration first.";
          return;
        }
        const { rotation, mirror } = detectRotationFromCSM(matrix);
        this.setPreviewTransform(rotation, mirror);
        this.detectMessage = `Preview set to ${rotation} degrees${mirror ? " + mirror" : ""}`;
      } catch {
        this.detectMessage = "Detection failed - is the camera_stage_mapping thing available?";
      } finally {
        this.detecting = false;
      }
    },
  },
};
</script>
