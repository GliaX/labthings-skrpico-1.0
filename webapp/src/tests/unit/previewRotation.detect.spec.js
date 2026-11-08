import { describe, it, expect } from "vitest";
import { detectRotationFromCSM } from "@/js_utils/previewRotation.mjs";

const LIVE = [
  [-1.987878173287538, 0.0531293079444194],
  [0.060263288913345786, 1.962400817209913],
];

describe("detectRotationFromCSM", () => {
  it("solves the live x-002 matrix to 90 degrees with mirror", () => {
    expect(detectRotationFromCSM(LIVE)).toEqual({ rotation: 90, mirror: true });
  });

  it("solves a slightly perturbed calibration to the same quadrant", () => {
    expect(
      detectRotationFromCSM([
        [-1.9, 0.12],
        [0.09, 1.9],
      ]),
    ).toEqual({ rotation: 90, mirror: true });
  });

  it("rejects no-calibration null", () => {
    expect(() => detectRotationFromCSM(null)).toThrow();
  });

  it("rejects singular matrices", () => {
    expect(() =>
      detectRotationFromCSM([
        [0, 0],
        [0, 0],
      ]),
    ).toThrow();
  });
});
