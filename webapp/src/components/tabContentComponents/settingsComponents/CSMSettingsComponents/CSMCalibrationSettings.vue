<template>
  <div id="CSMCalibrationSettings" ref="CSMCalibrationSettingsContainer" class="uk-width-large">
    <!--Show auto calibrate if default plugin is enabled-->
    <div v-if="'calibrate_xy' in actions" class="uk-margin-small">
      <action-button
        :button-primary="true"
        :can-terminate="true"
        :requires-confirmation="true"
        :confirmation-message="'Start recalibration of the stage to the camera? This may take a while, and the microscope will be locked during this time.'"
        thing="camera_stage_mapping"
        action="calibrate_xy"
        :submit-label="'Auto-Calibrate Using Camera'"
        :modal-progress="true"
        :stream-with-modal="true"
        @response="onRecalibrateResponse"
        @error="modalError"
      />
    </div>
    <button
      v-if="'last_calibration' in properties"
      v-show="showExtraSettings"
      type="button"
      class="uk-button uk-button-default uk-width-1-1"
      :disabled="!csmMatrix"
      :class="{ 'uk-disabled': !csmMatrix }"
      @click="getCalibrationData()"
    >
      Download Calibration Data
    </button>
    <div v-if="csmMatrix" style="margin: 10px">
      <details>
        <summary>Calibration Details</summary>
        <ul>
          <li>
            <strong>Current CSM Matrix:</strong>
            <br />
            <matrixDisplay :matrix="csmMatrix" />
          </li>
          <li>
            CSM calculated for images with a resolution of:
            <br />
            <matrixDisplay :matrix="csmResolution" :bracket-height="1.5" />
          </li>

          <li>Motor steps per pixel: {{ csmRatio }}</li>
          <li>
            Full field of view in motor steps:
            <br />
            <matrixDisplay :matrix="csmFOV" :bracket-height="1.5" />
          </li>
        </ul>
      </details>
    </div>
    <p v-else><strong>No Calibration Data Available</strong></p>
  </div>
</template>

<script>
import ActionButton from "@/components/labThingsComponents/actionButton.vue";
import matrixDisplay from "@/components/ui/matrixDisplay.vue";
import { useIntersectionObserver } from "@vueuse/core";

// Export main app
export default {
  name: "CSMCalibrationSettings",

  components: {
    ActionButton,
    matrixDisplay,
  },

  props: {
    showExtraSettings: {
      type: Boolean,
      required: false,
      default: true,
    },
  },

  emits: ["recalibrateResponse"],

  data() {
    return {
      csmMatrix: undefined,
      csmResolution: undefined,
      csmRatio: undefined,
      csmFOV: undefined,
    };
  },

  computed: {
    actions() {
      return this.thingDescription("camera_stage_mapping").actions;
    },
    properties() {
      return this.thingDescription("camera_stage_mapping").properties;
    },
  },

  mounted() {
    useIntersectionObserver(
      this.$refs.CSMCalibrationSettingsContainer,
      ([{ isIntersecting }]) => {
        this.visibilityChanged(isIntersecting);
      },
      { threshold: 0.0 },
    );
  },

  methods: {
    visibilityChanged(isVisible) {
      if (isVisible) {
        this.updateDisplayedCSM();
      }
    },
    getCalibrationData: async function () {
      try {
        let data = await this.readThingProperty("camera_stage_mapping", "last_calibration");
        if (data == {}) {
          throw "No calibration data available.";
        }
        const dataStr = JSON.stringify(data);
        const url = window.URL.createObjectURL(new Blob([dataStr]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", "csm_calibration.json");
        document.body.appendChild(link);
        link.click();
      } catch (error) {
        this.modalError(error); // Let mixin handle error
      }
    },
    updateDisplayedCSM: async function () {
      let csmMatrix = await this.readThingProperty(
        "camera_stage_mapping",
        "image_to_stage_displacement_matrix",
      );
      if (csmMatrix) {
        this.csmResolution = await this.readThingProperty(
          "camera_stage_mapping",
          "image_resolution",
        );
        csmMatrix[0][0] = Number(csmMatrix[0][0].toFixed(3));
        csmMatrix[0][1] = Number(csmMatrix[0][1].toFixed(3));
        csmMatrix[1][0] = Number(csmMatrix[1][0].toFixed(3));
        csmMatrix[1][1] = Number(csmMatrix[1][1].toFixed(3));
        this.csmMatrix = csmMatrix;
        this.csmRatio = Number((Math.abs(csmMatrix[1][0]) + Math.abs(csmMatrix[0][1])) / 2).toFixed(
          3,
        );
        // Note [1] then [0] as CSM reports resolution as (y, x)
        this.csmFOV = [
          Number((this.csmResolution[1] * this.csmRatio).toFixed(0)),
          Number((this.csmResolution[0] * this.csmRatio).toFixed(0)),
        ];
      }
    },
    onRecalibrateResponse: function (response) {
      this.modalNotify("Finished stage-to-camera calibration.");
      this.updateDisplayedCSM();
      this.$emit("recalibrateResponse", response);
    },
  },
};
</script>

<style lang="less">
.center-spinner {
  margin-left: auto;
  margin-right: auto;
}
</style>
