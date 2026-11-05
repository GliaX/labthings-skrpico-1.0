<template>
  <div id="cameraSettings">
    <div class="uk-grid uk-grid-divider uk-child-width-expand" uk-grid>
      <div class="uk-width-large">
        <h3>Automatic calibration</h3>
        <cameraCalibrationSettings :camera-uri="cameraUri" />
        <h3>Manual camera settings</h3>
        <div class="uk-margin-small-bottom">
          <server-specified-property-control
            v-for="(setting, index) in manualCameraSettings"
            :key="'cam_setting' + index"
            :property-data="setting"
          />
        </div>
      </div>

      <div id="mini-stream">
        <miniStreamDisplay :stream-id="setStreamId" />
      </div>
    </div>
  </div>
</template>

<script>
import cameraCalibrationSettings from "./cameraSettingsComponents/cameraCalibrationSettings.vue";
import miniStreamDisplay from "../../genericComponents/miniStreamDisplay.vue";
import ServerSpecifiedPropertyControl from "../../labThingsComponents/serverSpecifiedPropertyControl.vue";
import { mapState } from "pinia";
import { useSettingsStore } from "@/stores/settings.js";

// Export main app
export default {
  name: "CameraSettings",

  components: {
    cameraCalibrationSettings,
    miniStreamDisplay,
    ServerSpecifiedPropertyControl,
  },

  data() {
    return {
      manualCameraSettings: [],
      // This adds the parent name as value for prop streamId
      setStreamId: this.$options.name,
    };
  },

  computed: {
    ...mapState(useSettingsStore, ["baseUri"]),
    cameraUri() {
      return `${this.baseUri}/camera/`;
    },
  },

  async created() {
    this.manualCameraSettings = await this.readThingProperty("camera", "manual_camera_settings");
  },
};
</script>

<style lang="less">
#mini-stream {
  min-width: 300px;
  max-width: 600px;
  text-align: center;
  margin-left: auto;
  margin-right: auto;
  margin-top: 50px;
}
</style>
