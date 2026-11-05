<template>
  <div id="autofocusSettings">
    <h3>Objective</h3>
    <div class="uk-margin-small uk-form-stacked">
      <label class="uk-form-label" for="objectiveSelect">Mounted objective</label>
      <div class="uk-form-controls">
        <select
          id="objectiveSelect"
          v-model="objSel"
          class="uk-select uk-form-width-small"
          :disabled="objBusy"
          @change="writeObjective"
        >
          <option v-for="(label, val) in objectiveOptions" :key="val" :value="val">
            {{ label }}
          </option>
        </select>
      </div>
    </div>
    <p v-if="profileSummary" class="uk-text-meta uk-margin-small-top">
      Calibrated: {{ profileSummary }}
    </p>
    <h3>Focus profile calibration</h3>
    <p class="uk-text-meta">
      Bring the sample roughly into focus with the jog controls first - the calibration sweeps a
      small range around the current position and stores the sharpness curve for this objective.
    </p>
    <div class="uk-margin uk-width-medium">
      <action-button
        thing="autofocus"
        action="calibrate_focus_profile"
        :submit-data="{}"
        :submit-label="'Calibrate focus profile'"
        :button-primary="true"
        @finished="readProfile"
        @error="modalError"
      />
    </div>
  </div>
</template>

<script>
import ActionButton from "../../labThingsComponents/actionButton.vue";

export default {
  name: "AutofocusSettings",

  components: {
    ActionButton,
  },

  data() {
    return {
      objSel: "",
      objBusy: false,
      profile: null,
      objectiveOptions: {
        "": "Not set",
        "20x": "20x",
        "40x": "40x",
        "60x": "60x",
        "100x": "100x",
      },
    };
  },

  computed: {
    profileSummary: function () {
      const p = this.profile;
      if (!p || !p.fwhm_u) return "";
      let s = `fwhm ${Number(p.fwhm_u).toFixed(1)}u`;
      if (p.contrast) s += `, contrast ${Number(p.contrast).toFixed(2)}`;
      if (p.csm_scale) s += `, scale ${Number(p.csm_scale).toFixed(3)}`;
      return s;
    },
  },

  async created() {
    try {
      const v = await this.readThingProperty("autofocus", "objective");
      this.objSel = v ? String(v) : "";
    } catch {
      this.objSel = "";
    }
    this.readProfile();
  },

  methods: {
    writeObjective: async function () {
      this.objBusy = true;
      const value = this.objSel || null;
      try {
        await this.writeThingProperty("autofocus", "objective", value);
        this.readProfile();
      } catch (error) {
        this.modalError(error);
        this.readObjective();
      } finally {
        this.objBusy = false;
      }
    },
    readObjective: async function () {
      try {
        const v = await this.readThingProperty("autofocus", "objective");
        this.objSel = v ? String(v) : "";
      } catch {
        this.objSel = "";
      }
    },
    readProfile: async function () {
      try {
        const obj = await this.readThingProperty("autofocus", "objective");
        const profiles = await this.readThingProperty("autofocus", "objective_profiles");
        this.profile = profiles && obj ? profiles[obj] : null;
      } catch {
        this.profile = null;
      }
    },
  },
};
</script>

<style lang="less"></style>
