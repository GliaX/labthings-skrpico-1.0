<template>
  <div id="CSMCalibrationSettings" ref="CSMCalibrationSettingsContainer" class="uk-width-large">
    <!--Show auto calibrate if default plugin is enabled-->
    <div class="uk-margin-small">
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
    <div class="uk-margin-small">
      <button
        class="uk-button uk-button-primary uk-width-1-1"
        :disabled="detecting"
        @click="autoDetectView"
      >
        {{ detecting ? "Detecting… (stage will jog 4×)" : "Auto-Detect View Alignment" }}
      </button>
      <p v-if="detectResult" class="uk-text-meta uk-margin-remove">
        Measured: +X→({{ detectResult.vx_px[0] }},{{ detectResult.vx_px[1] }})px +Y→({{
          detectResult.vy_px[0]
        }},{{ detectResult.vy_px[1] }})px → stage→image rotation {{ stageToImageRotation }}°,
        mirror: {{ navigationMirror ? "yes" : "no" }}
      </p>
    </div>
    <button
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
import axios from "axios";
import { useIntersectionObserver } from "@vueuse/core";
import { useSettingsStore } from "@/stores/settings.js";
import { storeToRefs } from "pinia";
import labThingsMixin from "@/mixins/labThingsMixins.js";

// Export main app
export default {
  name: "CSMCalibrationSettings",

  components: {
    ActionButton,
    matrixDisplay,
  },
  mixins: [labThingsMixin],

  props: {
    showExtraSettings: {
      type: Boolean,
      required: false,
      default: true,
    },
  },

  emits: ["recalibrateResponse"],

  setup() {
    const store = useSettingsStore();
    const { navigationStepSize, navigationRotate, navigationMirror, stageToImageRotation } =
      storeToRefs(store);
    return {
      navigationStepSize,
      navigationRotate,
      navigationMirror,
      stageToImageRotation,
    };
  },

  data() {
    return {
      csmMatrix: undefined,
      csmResolution: undefined,
      csmRatio: undefined,
      csmFOV: undefined,
      detecting: false,
      detectResult: null,
    };
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
    async autoDetectView() {
      this.detecting = true;
      this.detectResult = null;
      try {
        const step = Math.max(
          Math.abs(this.navigationStepSize.x),
          Math.abs(this.navigationStepSize.y),
        );
        const response = await this.invokeAction(
          "camera_stage_mapping",
          "detect_view",
          { step: step },
          false,
        );
        const invocation = response.data;
        const url = invocation.href || invocation["@id"];
        const final = await this.pollUntilCompleteReturn(url);
        this.detectResult = final;
        this.stageToImageRotation = final.stage_to_image_rotation;
        this.navigationMirror = final.mirror;
        // Suggest a preview rotation so the view is upright relative to
        // the stage: preview = stageToImage + 180 keeps +X rightward.
        this.navigationRotate = (final.stage_to_image_rotation + 180) % 360;
      } catch (error) {
        this.modalError(error);
      } finally {
        this.detecting = false;
      }
    },
    async pollUntilCompleteReturn(url) {
      for (;;) {
        const response = await axios.get(url, { baseURL: this.baseUri });
        const status = response.data.status;
        if (status === "completed") {
          const output = await axios.get(`${url}/output`, {
            baseURL: this.baseUri,
          });
          return output.data;
        }
        if (status === "error") {
          throw new Error(
            (response.data.log || []).map((entry) => entry.message).join("; ") ||
              "detection failed",
          );
        }
        await new Promise((resolve) => setTimeout(resolve, 400));
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
