import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"


class ParcelProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = Field(None, pattern=NAME_PATTERN, description="folder name for the results")
    area_ha: float | None = Field(
        None, gt=0, description="area the caller expects; a mismatch is warned about"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="returned in result.json as is")
    highlight: dict[str, Any] | None = Field(
        None,
        description=(
            "GeoJSON Polygon or MultiPolygon (lng/lat) drawn as a thin outline on the maps, e.g. the field "
            "inside a larger analysed area; it does not change the analysis"
        ),
    )

    @model_validator(mode="after")
    def collect_extra(self) -> "ParcelProperties":
        if self.model_extra:
            self.metadata = {**self.model_extra, **self.metadata}
            self.model_extra.clear()
        return self


class ParcelRequest(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon or MultiPolygon, lng/lat (EPSG:4326)")
    properties: ParcelProperties = Field(default_factory=ParcelProperties)

    @classmethod
    def from_json(cls, text: str, *, default_name: str | None = None) -> "ParcelRequest":
        data = json.loads(text.lstrip("﻿"))
        if data.get("type") == "FeatureCollection":
            features = data.get("features") or []
            if len(features) != 1:
                raise ValueError(f"expected exactly one feature, got {len(features)}")
            data = features[0]
        if data.get("type") != "Feature":
            data = {"type": "Feature", "geometry": data, "properties": {}}
        data["properties"] = data.get("properties") or {}
        if default_name and not data["properties"].get("name"):
            data["properties"]["name"] = safe_name(default_name)
        return cls.model_validate(data)

    @property
    def name(self) -> str:
        if self.properties.name:
            return self.properties.name
        digest = hashlib.sha1(json.dumps(self.geometry, sort_keys=True).encode()).hexdigest()[:10]
        return f"parcel_{digest}"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")[:100]
    return cleaned or "parcel"
