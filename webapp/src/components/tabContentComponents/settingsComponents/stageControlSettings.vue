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
          <input v-model="stepSize.x" class="uk-input uk-form-small" type="number" />
        </div>
        <label class="uk-margin-small-right">
          <input v-model="invert.x" class="uk-checkbox" type="checkbox" />
          Invert x
        </label>
      </div>

      <div>
        <label class="uk-form-label" for="form-stacked-text">y</label>
        <div class="uk-form-controls">
          <input v-model="stepSize.y" class="uk-input uk-form-small" type="number" />
        </div>
        <label class="uk-margin-small-right">
          <input v-model="invert.y" class="uk-checkbox" type="checkbox" />
          Invert y
        </label>
      </div>

      <div>
        <label class="uk-form-label" for="form-stacked-text">z</label>
        <div class="uk-form-controls">
          <input v-model="stepSize.z" class="uk-input uk-form-small" type="number" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "StageControlSettings",

  computed: {
    // Note that as stepSize and invert are mutated (i.e. we change stepSize.x not stepSize)
    // rather than directly set we cannot use get() and set() computed to interact with the
    // store as changed won't be detected by set(). Instead use a deep watcher to
    // update the store (see ``watch:`` below)
    stepSize() {
      return this.$store.state.navigationStepSize;
    },
    invert() {
      return this.$store.state.navigationInvert;
    },
  },

  watch: {
    stepSize: {
      deep: true,
      handler(newVal) {
        this.$store.commit("changeNavigationStepSize", newVal);
      },
    },
    invert: {
      deep: true,
      handler(newVal) {
        this.$store.commit("changeNavigationInvert", newVal);
      },
    },
  },
};
</script>
