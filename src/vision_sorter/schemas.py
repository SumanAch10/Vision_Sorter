"""Public API contract and cross-field safety invariants."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class Point(BaseModel):
    x: Finite
    y: Finite


class Mask(BaseModel):
    encoding: Literal["rle-start-length"]
    size: tuple[int, int]
    runs: list[tuple[int, int]]

    @model_validator(mode="after")
    def aligned_runs(self):
        w, h = self.size
        if w < 1 or h < 1:
            raise ValueError("Invalid mask dimensions")
        end = 0
        for start, length in self.runs:
            if start < end or length < 1 or start + length > w * h:
                raise ValueError("Invalid or overlapping mask runs")
            end = start + length
        return self


class Candidate(BaseModel):
    object_class: str
    confidence: Probability


class Step(BaseModel):
    action: Literal["approach", "pick", "place", "return_home"]
    position_mm: Point | None = None
    destination: str | None = None


class Simulation(BaseModel):
    executed: bool
    mode: Literal["simulation"]
    steps: list[Step]
    reason: str | None = None


class Segmentation(BaseModel):
    model_version: str
    semantic_classes: list[str]
    score: Probability
    score_kind: str
    foreground_area_fraction: Probability
    component_count: int = Field(ge=0)


class BoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    request_id: str
    status: Literal["ready", "needs_review"]
    object_class: str
    confidence: Probability
    confidence_kind: str
    confidence_checks_passed: bool
    classification_status: Literal["uncertain", "confident"] | None = None
    candidates: list[Candidate] | None = None
    confidence_margin: Probability | None = None
    warnings: list[str]
    center_px: Point | None
    pick_point_mm: Point | None
    destination: str | None
    model_version: str
    demo: bool
    reasons: list[str]
    image_size: tuple[int, int]
    mask: Mask
    mask_source: str | None = None
    bbox_px: BoundingBox | None = None
    segmentation: Segmentation | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    simulation: Simulation

    @model_validator(mode="after")
    def safe_decision(self):
        if self.image_size != self.mask.size:
            raise ValueError("Mask and image dimensions differ")
        w, h = self.image_size
        if self.center_px and not (0 <= self.center_px.x < w and 0 <= self.center_px.y < h):
            raise ValueError("Center lies outside the image")
        if self.bbox_px and (
            self.bbox_px.x + self.bbox_px.width > w or self.bbox_px.y + self.bbox_px.height > h
        ):
            raise ValueError("Bounding box lies outside the image")
        if self.status == "needs_review" and (
            self.pick_point_mm is not None
            or self.destination is not None
            or self.simulation.executed
            or self.simulation.steps
        ):
            raise ValueError("Review decisions cannot issue a pick")
        if self.status == "ready" and (
            self.reasons
            or self.pick_point_mm is None
            or not self.destination
            or not self.simulation.executed
        ):
            raise ValueError("Ready decision lacks validated command fields")
        return self


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    retryable: bool = False
