from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from argo_geoai.parcels import Parcel
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message


@dataclass
class RunContext:
    parcel: Parcel
    workdir: Path
    output_root: Path
    vegetation_result: dict[str, Any] | None = None


@dataclass
class ToolOutput[MetricsT: BaseModel]:
    metrics: MetricsT
    summary: list[Finding] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    data_sources: list[DataSource] = field(default_factory=list)
    warnings: list[Message] = field(default_factory=list)
    partial: bool = False


class Tool[ParamsT: BaseModel, MetricsT: BaseModel](ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    title: ClassVar[Label]
    maps_title: ClassVar[str] = "Карти ділянки"
    params_model: type[ParamsT]
    metrics_model: type[MetricsT]

    @abstractmethod
    def run(self, params: ParamsT, context: RunContext) -> ToolOutput[MetricsT]: ...
