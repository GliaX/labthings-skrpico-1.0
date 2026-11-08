import { describe, it, expect, beforeEach } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import piniaPluginPersistedstate from "pinia-plugin-persistedstate";
import { useSettingsStore } from "@/stores/settings.js";

describe("settings store preview transform", () => {
  beforeEach(() => {
    setActivePinia(createPinia().use(piniaPluginPersistedstate));
    localStorage.clear();
  });

  it("defaults to 0 degrees, unmirrored", () => {
    const store = useSettingsStore();
    expect(store.previewRotation).toBe(0);
    expect(store.previewMirror).toBe(false);
  });

  it("setPreviewTransform stores rotation and mirror together", () => {
    const store = useSettingsStore();
    store.setPreviewTransform(270, true);
    expect(store.previewRotation).toBe(270);
    expect(store.previewMirror).toBe(true);
  });

  it.each([45, -90, 360, 91, "90", null])("rejects invalid angle %s", (angle) => {
    const store = useSettingsStore();
    expect(() => store.setPreviewTransform(angle, false)).toThrow(RangeError);
    expect(store.previewRotation).toBe(0);
    expect(store.previewMirror).toBe(false);
  });

  it("leaves navigation defaults unchanged", () => {
    const store = useSettingsStore();
    expect(store.navigationStepSize).toEqual({ x: 200, y: 200, z: 50 });
    expect(store.navigationInvert).toEqual({ x: false, y: false, z: false });
  });
});
