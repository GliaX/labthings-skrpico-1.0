<template>
  <!-- Grid managing tab content -->
  <div uk-grid class="uk-height-1-1 uk-margin-remove uk-padding-remove">
    <div class="settings-nav">
      <ul class="uk-nav uk-nav-default">
        <li class="uk-nav-header">Application Settings</li>
        <li v-for="item in appTabs" :key="'setting-' + item.id + '-tab-content'">
          <tabIcon
            :id="'setting-' + item.id + '-tab-content'"
            :tab-i-d="item.id"
            :show-title="false"
            :show-tooltip="false"
            :require-connection="item.requireConnection"
            :current-tab="currentTab"
            @set-tab="setTab"
          >
            {{ item.title }}
          </tabIcon>
        </li>
        <li class="uk-nav-header">Microscope Settings</li>
        <button
          submit-label="Launch Calibration Wizard"
          type="button"
          class="uk-button uk-button-default uk-width-1-1"
          @click="$emit('startCalModal')"
        >
          Launch Calibration Wizard
        </button>
        <li v-for="item in calibrationTabs" :key="'setting-' + item.id + '-tab-icon'">
          <tabIcon
            :id="'setting-' + item.id + '-tab-icon'"
            :tab-i-d="item.id"
            :show-title="false"
            :show-tooltip="false"
            :require-connection="item.requireConnection"
            :current-tab="currentTab"
            @set-tab="setTab"
          >
            {{ item.title }}
          </tabIcon>
        </li>
      </ul>
    </div>
    <div class="view-component uk-width-expand uk-padding-small">
      <tabContent
        v-for="item in allTabs"
        :id="'setting-' + item.id + '-tab-content'"
        :key="'setting-' + item.id + '-tab-content'"
        :tab-i-d="item.id"
        :require-connection="item.requireConnection"
        :current-tab="currentTab"
      >
        <component :is="item.component"></component>
      </tabContent>
    </div>
  </div>
</template>

<script>
import displaySettings from "./settingsComponents/displaySettings.vue";
import stageControlSettings from "./settingsComponents/stageControlSettings.vue";
import cameraSettings from "./settingsComponents/cameraSettings.vue";
import autofocusSettings from "./settingsComponents/autofocusSettings.vue";
import CSMSettings from "./settingsComponents/CSMSettings.vue";
import stageSettings from "./settingsComponents/stageSettings.vue";
// Import generic components
import tabIcon from "../genericComponents/tabIcon.vue";
import tabContent from "../genericComponents/tabContent.vue";
import { markRaw } from "vue";

// Export main app
export default {
  name: "SettingsContent",

  components: {
    tabIcon,
    tabContent,
  },

  emits: ["startCalModal"],

  data: function () {
    return {
      selected: "display",
      currentTab: "display",
      coreAppTabs: [
        {
          id: "display",
          title: "Display",
          requireConnection: false,
          component: markRaw(displaySettings),
          requiredThings: [],
        },
        {
          id: "stage-control",
          title: "Stage Control Preferences",
          requireConnection: false,
          component: markRaw(stageControlSettings),
          requiredThings: ["stage"],
        },
      ],
      coreCalibrationTabs: [
        {
          id: "camera",
          title: "Camera",
          requireConnection: true,
          component: markRaw(cameraSettings),
          requiredThings: [],
        },
        {
          id: "autofocus",
          title: "Autofocus",
          requireConnection: true,
          component: markRaw(autofocusSettings),
          requiredThings: ["autofocus"],
        },
        {
          id: "stage",
          title: "Stage",
          requireConnection: true,
          component: markRaw(stageSettings),
          requiredThings: ["stage"],
        },
        {
          id: "mapping",
          title: "Camera to Stage Mapping",
          requireConnection: true,
          component: markRaw(CSMSettings),
          requiredThings: ["camera_stage_mapping"],
        },
      ],
    };
  },

  computed: {
    allTabs() {
      return [...this.appTabs, ...this.calibrationTabs];
    },
    appTabs() {
      // Filter core top tabs based on available Things.
      return this.coreAppTabs.filter((tab) => {
        if (!tab.requiredThings || tab.requiredThings.length === 0) return true;
        return tab.requiredThings.every((thing) => this.thingAvailable(thing));
      });
    },
    calibrationTabs() {
      // Filter core top tabs based on available Things.
      return this.coreCalibrationTabs.filter((tab) => {
        if (!tab.requiredThings || tab.requiredThings.length === 0) return true;
        return tab.requiredThings.every((thing) => this.thingAvailable(thing));
      });
    },
  },
  methods: {
    setTab: function (event, tab) {
      if (!(this.currentTab == tab)) {
        this.currentTab = tab;
      }
    },
  },
};
</script>

<style lang="less" scoped>
// Custom UIkit CSS modifications
@import url("../../assets/less/variables.less");

.settings-nav {
  overflow: hidden auto;
  width: 250px;
  padding: 10px;
  background-color: rgba(180, 180, 180, 0.03);
  border-width: 0 1px 0 0;
  border-style: solid;
  border-color: rgba(180, 180, 180, 0.25);
}

.settings-nav li > a {
  padding-left: 6px !important;
  border-radius: @button-border-radius;
}
</style>
