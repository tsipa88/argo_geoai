from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "3.0"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Label(Model):
    """A short value shown to people: the same meaning in English and Ukrainian."""

    en: str
    uk: str


class RunStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"

    @property
    def label(self) -> Label:
        return _STATUS_LABELS[self]


_STATUS_LABELS = {
    RunStatus.OK: Label(en="ok", uk="успішно"),
    RunStatus.PARTIAL: Label(en="partial", uk="частково"),
    RunStatus.ERROR: Label(en="error", uk="помилка"),
}


class ToolInfo(Model):
    name: str
    version: str
    title: Label


class RunInfo(Model):
    id: UUID
    started_at: datetime
    finished_at: datetime
    duration_s: float
    status: RunStatus
    status_label: Label
    argo_geoai_version: str


class ParcelInfo(Model):
    name: str = Field(description="from the request, or derived from the geometry")
    parts: int = Field(description="separate parts of the parcel (MultiPolygon)")
    area_ha: float = Field(description="area of the polygon")
    expected_area_ha: float | None = Field(description="area the caller expected, from the request")
    bbox: tuple[float, float, float, float] = Field(
        description="min lng, min lat, max lng, max lat (EPSG:4326)"
    )
    centroid: tuple[float, float] = Field(description="lng, lat (EPSG:4326)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="returned from the request as is")


class DataSource(Model):
    collection: str
    title: str
    provider: str
    resolution_m: float
    items: list[str] = Field(default_factory=list)


class Finding(Model):
    """One human-readable conclusion produced by a fixed rule (never free-form reasoning)."""

    code: str
    level: Literal["info", "notice", "warning"]
    text: str = Field(description="Ukrainian sentence for people")


class Artifact(Model):
    path: str = Field(description="relative to the folder of result.json")
    media_type: str
    description: str = Field(description="Ukrainian description for people")


class Message(Model):
    code: str
    text: str


class ToolResult[MetricsT: BaseModel, ParamsT: BaseModel](Model):
    schema_version: str = SCHEMA_VERSION
    tool: ToolInfo
    run: RunInfo
    parcel: ParcelInfo
    parameters: ParamsT
    data_sources: list[DataSource] = Field(default_factory=list)
    summary: list[Finding] = Field(default_factory=list)
    metrics: MetricsT | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[Message] = Field(default_factory=list)
    errors: list[Message] = Field(default_factory=list)

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
