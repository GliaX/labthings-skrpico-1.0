import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { eventBus } from "@/eventBus.js";
import StreamDisplay from "@/components/tabContentComponents/streamContent.vue";

const NATURAL = { naturalWidth: 800, naturalHeight: 600 };
const RECT = { left: 0, top: 0, width: 400, height: 300 };

function clickEvent(dx, dy) {
  return {
    clientX: RECT.left + RECT.width / 2 + dx,
    clientY: RECT.top + RECT.height / 2 + dy,
    target: {
      ...NATURAL,
      offsetWidth: RECT.width,
      offsetHeight: RECT.height,
      getBoundingClientRect: () => ({ ...RECT }),
    },
  };
}

async function mountStream(previewRotation, previewMirror = false) {
  return mount(StreamDisplay, {
    props: { streamId: "testProp" },
    global: {
      plugins: [
        createTestingPinia({
          initialState: {
            settings: {
              ready: true,
              disableStream: false,
              waiting: false,
              baseUri: "http://microscope-test.local:5000/api/v3",
              previewRotation,
              previewMirror,
            },
          },
        }),
      ],
    },
  });
}

const SCALE = Math.max(NATURAL.naturalWidth / RECT.width, NATURAL.naturalHeight / RECT.height);

function legacyPayload(dx, dy) {
  return { x: dx * SCALE, y: dy * SCALE };
}

describe("streamContent click-to-move under preview transform", () => {
  let emitSpy;

  beforeEach(() => {
    emitSpy = vi.spyOn(eventBus, "emit");
    emitSpy.mockClear();
  });

  it("at rotation 0 unmirrored emits the legacy payload unchanged", async () => {
    const wrapper = await mountStream(0, false);
    wrapper.vm.clickMonitor(clickEvent(-100, 50));
    expect(emitSpy).toHaveBeenCalledWith("globalMoveInImageCoordinatesEvent", legacyPayload(-100, 50));
  });

  it("at 90 mirrored, inverse-transforms the screen click into raw image coordinates", async () => {
    const wrapper = await mountStream(90, true);
    wrapper.vm.clickMonitor(clickEvent(-100, 50));
    const rx = -100 * 0 + 50 * 1;
    const ry = 100 * 1 + 50 * 0;
    expect(emitSpy).toHaveBeenCalledWith("globalMoveInImageCoordinatesEvent", {
      x: -rx * SCALE,
      y: ry * SCALE,
    });
  });

  it("clicking the centre emits zero motion at every transform", async () => {
    for (const rotation of [0, 90, 180, 270]) {
      for (const mirror of [false, true]) {
        emitSpy.mockClear();
        const wrapper = await mountStream(rotation, mirror);
        wrapper.vm.clickMonitor(clickEvent(0, 0));
        expect(emitSpy).toHaveBeenCalledWith("globalMoveInImageCoordinatesEvent", { x: 0, y: 0 });
        wrapper.unmount();
      }
    }
  });

  it("mirror-only transform negates the horizontal component of the click", async () => {
    const wrapper = await mountStream(0, true);
    wrapper.vm.clickMonitor(clickEvent(-100, 50));
    expect(emitSpy).toHaveBeenCalledWith("globalMoveInImageCoordinatesEvent", {
      x: 100 * SCALE,
      y: 50 * SCALE,
    });
  });
});
