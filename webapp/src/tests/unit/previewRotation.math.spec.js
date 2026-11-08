import { describe, it, expect } from "vitest";
import {
  assertValidRotationAngle,
  applyRotationToStageVector,
  previewCssTransform,
  displayedOffsetToRawOffset,
} from "@/js_utils/previewRotation.mjs";

describe("applyRotationToStageVector", () => {
  it.each([
    [1, 0, 0, { x: 1, y: 0 }],
    [1, 0, 90, { x: 0, y: 1 }],
    [1, 0, 180, { x: -1, y: 0 }],
    [1, 0, 270, { x: 0, y: -1 }],
    [0, 1, 0, { x: 0, y: 1 }],
    [0, 1, 90, { x: -1, y: 0 }],
    [0, 1, 180, { x: 0, y: -1 }],
    [0, 1, 270, { x: 1, y: 0 }],
  ])("maps (%i, %i) at %i degrees", (x, y, deg, expected) => {
    expect(applyRotationToStageVector(x, y, deg)).toEqual(expected);
  });

  it("rejects invalid angles", () => {
    expect(() => assertValidRotationAngle(45)).toThrow(RangeError);
    expect(() => applyRotationToStageVector(1, 0, 91)).toThrow();
  });
});

describe("previewCssTransform", () => {
  it.each([
    [0, false, "rotate(0deg)"],
    [90, false, "rotate(90deg)"],
    [270, true, "rotate(270deg) scaleX(-1)"],
  ])("builds the CSS transform for %i deg mirror=%s", (deg, mirror, expected) => {
    expect(previewCssTransform(deg, mirror)).toBe(expected);
  });
});

describe("displayedOffsetToRawOffset", () => {
  it("is identity at 0 degrees without mirror", () => {
    expect(displayedOffsetToRawOffset(120, -80, 0, false)).toEqual({ x: 120, y: -80 });
  });

  it("mirroring flips only the x component", () => {
    expect(displayedOffsetToRawOffset(120, -80, 0, true)).toEqual({ x: -120, y: -80 });
  });

  it("undoes the preview rotation of a raw offset", () => {
    const raw = { x: 200, y: -100 };
    const rad = (90 * Math.PI) / 180;
    const shown = { x: raw.x * Math.cos(rad) - raw.y * Math.sin(rad), y: raw.x * Math.sin(rad) + raw.y * Math.cos(rad) };
    const back = displayedOffsetToRawOffset(shown.x, shown.y, 90, false);
    expect(back.x).toBeCloseTo(raw.x, 9);
    expect(back.y).toBeCloseTo(raw.y, 9);
  });

  it("round-trips at 180 with mirror", () => {
    const raw = { x: 30, y: 40 };
    const once = displayedOffsetToRawOffset(raw.x, raw.y, 180, true);
    const twice = displayedOffsetToRawOffset(once.x, once.y, 180, true);
    expect(twice.x).toBeCloseTo(raw.x, 9);
    expect(twice.y).toBeCloseTo(raw.y, 9);
  });
});
