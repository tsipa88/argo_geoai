from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from argo_geoai.config import get_settings
from argo_geoai.parcels import from_request
from argo_geoai.request import ParcelRequest
from argo_geoai.runner import run_tool
from argo_geoai.schema import ToolResult
from argo_geoai.tools import TOOLS


@dataclass(frozen=True)
class RunOutcome:
    tool: str
    result: ToolResult[Any, Any]
    folder: Path


def analyze(
    request: ParcelRequest,
    *,
    tools: Sequence[str] | None = None,
    params: Mapping[str, BaseModel] | None = None,
    output_root: Path | None = None,
) -> list[RunOutcome]:
    requested = list(TOOLS) if tools is None else list(tools)
    unknown = sorted(set(requested) - set(TOOLS))
    if unknown:
        raise ValueError(f"unknown tools: {', '.join(unknown)}; available: {', '.join(TOOLS)}")

    parcel = from_request(request)
    root = output_root or get_settings().output_dir
    outcomes = []
    for name in (name for name in TOOLS if name in requested):
        tool = TOOLS[name]
        tool_params = (params or {}).get(name) or tool.params_model()
        result, folder = run_tool(tool, parcel, tool_params, root)
        outcomes.append(RunOutcome(tool=name, result=result, folder=folder))
    return outcomes
