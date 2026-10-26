<template>
  <div id="modal-example" ref="calibrationModalEl" uk-modal="bg-close: false; esc-close:false;">
    <div class="uk-modal-dialog uk-modal-body">
      <!-- Get the style from uk-close, but use stop.prevent to avoid it actually closing -->
      <button
        type="button"
        class="uk-modal-close-default"
        uk-close
        @click.stop.prevent="confirmClose"
      ></button>
      <template v-if="onWelcomePage">
        <img src="/logo_colour_curves.svg" alt="OpenFlexure Microscope" />
        <h1>Welcome to your microscope</h1>
      </template>
      <h2 v-else class="uk-modal-title">Microscope Calibration</h2>

      <component
        v-bind="currentTask.props"
        :is="currentTask.component"
        v-if="currentTask"
        :key="taskIndex"
        :first="isFirstTask"
        :final="isFinalTask"
        :start-on-last="movingBackward"
        @next="nextTask"
        @back="previousTask"
      />
    </div>
  </div>
</template>

<script>
import singleStepTask from "./calibrationWizardComponents/singleStepTask.vue";
import welcomeStep from "./calibrationWizardComponents/welcomeStep.vue";
import zMotorDirectionStep from "./calibrationWizardComponents/zMotorDirectionStep.vue";
import strobeFlashStep from "./calibrationWizardComponents/strobeFlashStep.vue";
import cameraCalibrationTask from "./calibrationWizardComponents/cameraCalibrationTask.vue";
import cameraStageMappingTask from "./calibrationWizardComponents/cameraStageMappingTask.vue";
import finalStep from "./calibrationWizardComponents/finalStep.vue";
import { markRaw } from "vue";

export default {
  name: "CalibrationWizard",

  components: {},

  emits: ["onClose"],

  data: function () {
    return {
      isNeeded: undefined,
      includeWelcome: false,
      availableCalibrationTasks: {},
      tasks: [],
      taskIndex: 0,
      movingBackward: false,
      modalOpen: false,
      confirmingClose: false,
    };
  },

  computed: {
    onWelcomePage() {
      return this.includeWelcome && this.taskIndex === 0;
    },
    currentTask() {
      return this.tasks[this.taskIndex] || null;
    },
    isFirstTask() {
      return this.taskIndex === 0;
    },
    isFinalTask() {
      return this.taskIndex === this.tasks.length - 1;
    },
  },

  mounted() {
    this.$refs["calibrationModalEl"].addEventListener("hidden", this.onHide);
    // Start discovering the available calibration tasks.
    // Methods that depend on this must await this.calibrationTasksReady
    // before accessing this.availableCalibrationTasks.
    // This is a Promise, which will be resolved once the initial read is complete
    this.calibrationTasksReady = this.initialiseCalibrationTasks();
  },

  methods: {
    /**
     * Check all calibratable Things to see which require calibration.
     *
     * Iterates over `this.availableCalibrationTasks` (set during mounted()) and reads the
     * `calibration_required` property.
     *
     * Returns a list of thing names that report calibration is required.
     */
    async check_things_needing_calibration() {
      const needsCalibration = [];
      const calibrateableThings = Object.keys(this.availableCalibrationTasks);

      for (const name of calibrateableThings) {
        const properties = this.thingDescription(name)?.properties ?? {};
        if ("calibration_required" in properties) {
          const thingNeedsCal = await this.readThingProperty(name, "calibration_required");
          if (thingNeedsCal) {
            needsCalibration.push(name);
          }
        }
      }
      return needsCalibration;
    },

    /**
     * Get a list of Things which have tasks which can be calibrated
     */
    async initialiseCalibrationTasks() {
      const allCalibrationTasks = {
        stage: {
          component: markRaw(singleStepTask),
          props: {
            stepComponent: zMotorDirectionStep,
            title: "Stage Calibration",
          },
        },
        camera: {
          component: markRaw(cameraCalibrationTask),
          props: {},
        },
        camera_stage_mapping: {
          component: markRaw(cameraStageMappingTask),
          props: {},
        },
      };

      const availableTasks = Object.fromEntries(
        Object.entries(allCalibrationTasks).filter(([thing]) => this.thingAvailable(thing)),
      );

      const calibratableTasks = {};

      for (const [thing, taskComponent] of Object.entries(availableTasks)) {
        const properties = this.thingDescription(thing)?.properties ?? {};

        // If the Thing exposes can_calibrate, check its current value.
        // Otherwise, assume it can be calibrated.
        const canCalibrate =
          "can_calibrate" in properties
            ? await this.readThingProperty(thing, "can_calibrate")
            : true;

        if (canCalibrate) {
          calibratableTasks[thing] = taskComponent;
        }
      }

      this.availableCalibrationTasks = calibratableTasks;
    },

    resetData: function () {
      this.movingBackward = false;
      this.taskIndex = 0;
    },

    /**
     * Create the calibration wizard task list dynamically.
     */
    create_task_list(thingsToCal, includeWelcome = true) {
      //Set includeWelcome so it is possible to check if on the welcome page.
      this.includeWelcome = includeWelcome;
      const tasks = [];

      // Illumination flash first: later steps depend on working
      // illumination, so calibrate it before anything else happens.
      if (this.thingAvailable("strobe_control")) {
        tasks.push({
          component: markRaw(singleStepTask),
          props: {
            stepComponent: markRaw(strobeFlashStep),
            title: "Illumination Flash",
          },
        });
      }

      // Optionally include the welcome screen
      if (includeWelcome) {
        tasks.push({
          component: markRaw(singleStepTask),
          props: {
            stepComponent: markRaw(welcomeStep),
          },
        });
      }

      // Add calibration task for each thing
      for (const thing of thingsToCal) {
        tasks.push(this.availableCalibrationTasks[thing]);
      }

      // Always include the final step
      tasks.push({
        component: markRaw(singleStepTask),
        props: {
          stepComponent: markRaw(finalStep),
        },
      });

      this.tasks = tasks;
    },

    /**
     *Check if the calibration modal is needed, and only show it if it is.
     */
    show_if_needed: async function () {
      // Await calibrationTasksReady, this promise resolves when the tasks that can be
      // calibrated has been read from the server.
      await this.calibrationTasksReady;
      const thingsToCal = await this.check_things_needing_calibration();

      // Check if this calibration wizard can actually do anything useful
      if (thingsToCal.length > 0) {
        this.resetData();
        this.create_task_list(thingsToCal);
        this.show();
      } else {
        // If not needed, we just return the onClose event immediately
        this.onHide();
      }
    },

    // Forces modal to show on button press
    force_show: function () {
      const allThings = Object.keys(this.availableCalibrationTasks);

      this.resetData();
      this.create_task_list(allThings, false);
      this.show();
    },

    show: function () {
      this.modalOpen = true;
      window.addEventListener("keydown", this.handleKeydown);

      // Show the modal element
      var el = this.$refs["calibrationModalEl"];
      this.showModalElement(el); // Calls the mixin
    },

    hide: function () {
      this.modalOpen = false;
      window.removeEventListener("keydown", this.handleKeydown);

      // Show the modal
      var el = this.$refs["calibrationModalEl"];
      this.hideModalElement(el); // Calls the mixin
    },

    onHide: function () {
      this.$emit("onClose");
    },

    handleKeydown(event) {
      if (event.key === "Escape" && !event.repeat && this.modalOpen && !this.confirmingClose) {
        event.preventDefault();
        this.confirmClose();
      }
    },

    confirmClose() {
      if (this.confirmingClose) return;
      this.confirmingClose = true;
      let confirmationMessage =
        "Close calibration wizard?<br><br>This can be re-opened from the Settings tab at any time.";
      // Use standard modal confirmation
      this.modalConfirm(confirmationMessage)
        .then(
          () => {
            // User clicked YES → hide modal
            this.hide();
          },
          () => {
            // User clicked NO → do nothing, modal stays open
          },
        )
        .finally(() => {
          this.confirmingClose = false; // reset flag when confirmation modal is gone
        });
    },

    /*
     * Move to the previous task.
     */
    previousTask: function () {
      this.movingBackward = true;
      if (this.taskIndex > 0) {
        this.taskIndex = this.taskIndex - 1;
      }
    },

    /*
     * Move to the next task or close the modal if this is the final task.
     */
    nextTask: function () {
      this.movingBackward = false;
      if (this.taskIndex < this.tasks.length - 1) {
        this.taskIndex = this.taskIndex + 1;
        return true;
      } else {
        this.hide();
      }
    },
  },
};
</script>
