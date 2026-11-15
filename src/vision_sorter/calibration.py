"""Planar projective calibration; no robot-specific movement lives here."""

import json
from pathlib import Path

import numpy as np


class Calibration:
    def __init__(self, value):
        self.value = value
        self.matrix = np.asarray(value["homography"], dtype=float)
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or abs(np.linalg.det(self.matrix)) < 1e-12
        ):
            raise ValueError("Calibration must contain a finite, invertible 3x3 homography")
        self.size = value["image_size"]
        self.bounds = value["workspace_mm"]
        if len(self.size) != 2 or min(self.size) <= 0 or len(self.bounds) != 4:
            raise ValueError("Invalid calibration dimensions")
        if (
            not np.isfinite(self.bounds).all()
            or self.bounds[2] <= self.bounds[0]
            or self.bounds[3] <= self.bounds[1]
        ):
            raise ValueError("Invalid workspace bounds")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def project(self, x, y, size):
        if list(size) != self.size:
            raise ValueError("calibration_resolution_mismatch")
        if self.value.get("validated") is not True:
            raise ValueError("calibration_not_validated")
        error = self.value.get("rms_error_mm")
        if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 5:
            raise ValueError("calibration_error_too_large")
        point = self.matrix @ [x, y, 1]
        if abs(point[2]) < 1e-9:
            raise ValueError("invalid_projection")
        px, py = point[:2] / point[2]
        left, top, right, bottom = self.bounds
        if not (left <= px <= right and top <= py <= bottom):
            raise ValueError("outside_workspace")
        return {"x": round(float(px), 3), "y": round(float(py), 3)}


def fit_calibration(pixels, millimeters, image_size, workspace):
    a = np.asarray(pixels, dtype=float)
    b = np.asarray(millimeters, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[1] != 2
        or len(a) < 4
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Provide at least four finite corresponding 2D points")
    rows, targets = [], []
    for (x, y), (u, v) in zip(a, b):
        rows.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        targets.extend([u, v])
    if np.linalg.matrix_rank(rows) < 8:
        raise ValueError("Calibration points are degenerate")
    h = np.append(np.linalg.lstsq(rows, targets, rcond=None)[0], 1).reshape(3, 3)
    projected = np.c_[a, np.ones(len(a))] @ h.T
    projected = projected[:, :2] / projected[:, 2:]
    rms = float(np.sqrt(np.mean(np.sum((projected - b) ** 2, axis=1))))
    value = dict(
        version="table-v1",
        image_size=image_size,
        workspace_mm=workspace,
        homography=h.tolist(),
        validated=False,
        rms_error_mm=rms,
    )
    Calibration(value)
    return value
