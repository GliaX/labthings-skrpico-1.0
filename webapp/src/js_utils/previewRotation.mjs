export const VALID_ROTATIONS = [0, 90, 180, 270];

export function assertValidRotationAngle(angleDeg) {
  if (!VALID_ROTATIONS.includes(angleDeg)) {
    throw new RangeError(
      `Invalid rotation angle: ${angleDeg}. Must be one of ${VALID_ROTATIONS.join(", ")}.`,
    );
  }
}

export function rotationAngleToMatrix(angleDeg) {
  assertValidRotationAngle(angleDeg);
  switch (angleDeg) {
    case 0:
      return [
        [1, 0],
        [0, 1],
      ];
    case 90:
      return [
        [0, -1],
        [1, 0],
      ];
    case 180:
      return [
        [-1, 0],
        [0, -1],
      ];
    case 270:
      return [
        [0, 1],
        [-1, 0],
      ];
  }
}

export function applyRotationToStageVector(x, y, angleDeg) {
  const m = rotationAngleToMatrix(angleDeg);
  return {
    x: x * m[0][0] + y * m[0][1] + 0,
    y: x * m[1][0] + y * m[1][1] + 0,
  };
}

export function rotatePreviewPoint(px, py, width, height, angleDeg) {
  const centreX = width / 2;
  const centreY = height / 2;
  const rotated = applyRotationToStageVector(px - centreX, py - centreY, angleDeg);
  return { x: rotated.x + centreX, y: rotated.y + centreY };
}

export function previewCssTransform(rotationDeg, mirror) {
  assertValidRotationAngle(rotationDeg);
  return mirror ? `rotate(${rotationDeg}deg) scaleX(-1)` : `rotate(${rotationDeg}deg)`;
}

function snapCardinal(value) {
  if (Math.abs(value) < 1e-12) {
    return 0;
  }
  if (Math.abs(value - 1) < 1e-12) {
    return 1;
  }
  if (Math.abs(value + 1) < 1e-12) {
    return -1;
  }
  return value;
}

export function displayedOffsetToRawOffset(dx, dy, rotationDeg, mirror) {
  const rad = (rotationDeg * Math.PI) / 180;
  const c = snapCardinal(Math.cos(rad));
  const s = snapCardinal(Math.sin(rad));
  const rx = dx * c + dy * s;
  const ry = -dx * s + dy * c;
  const x = mirror ? -rx : rx;
  return { x: x + 0, y: ry + 0 };
}

function rawFromDisplayedForAxisSearch(dx, dy, rotationDeg) {
  const u = dx;
  const v = dy;
  const rad = (rotationDeg * Math.PI) / 180;
  const c = snapCardinal(Math.cos(rad));
  const s = snapCardinal(Math.sin(rad));
  return { x: u * c + v * s + 0, y: -u * s + v * c + 0 };
}

function stageVectorForDisplayedOffset(matrix, dx, dy, rotation) {
  const raw = rawFromDisplayedForAxisSearch(dx, dy, rotation);
  return {
    sx: raw.y * matrix[0][0] + raw.x * matrix[0][1],
    sy: raw.y * matrix[1][0] + raw.x * matrix[1][1],
  };
}

export function detectRotationFromCSM(matrix, angleToleranceDeg = 5) {
  if (
    !Array.isArray(matrix) ||
    matrix.length !== 2 ||
    matrix.some((row) => !Array.isArray(row) || row.length !== 2)
  ) {
    throw new TypeError("CSM matrix must be a 2x2 array");
  }
  const det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  if (!det) {
    throw new TypeError("CSM matrix is singular");
  }
  const cosTol = Math.cos((angleToleranceDeg * Math.PI) / 180);
  let rotation = null;
  for (const candidate of [0, 90, 180, 270]) {
    const right = stageVectorForDisplayedOffset(matrix, 1, 0, candidate);
    const up = stageVectorForDisplayedOffset(matrix, 0, 1, candidate);
    const cosRight = right.sx / (Math.hypot(right.sx, right.sy) || 1);
    const cosUp = up.sy / (Math.hypot(up.sx, up.sy) || 1);
    if (cosRight >= cosTol && cosUp >= cosTol) {
      rotation = candidate;
      break;
    }
  }
  if (rotation === null) {
    throw new TypeError("CSM matrix is not close to a quarter-turn mount");
  }
  return { rotation, mirror: det < 0 };
}
