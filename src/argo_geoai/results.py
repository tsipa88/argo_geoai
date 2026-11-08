import json
from datetime import datetime
from pathlib import Path
from typing import Any

from argo_geoai.parcels import Parcel
from argo_geoai.schema import RunStatus

RESULT_FILE = "result.json"


def run_folder(output_root: Path, parcel: Parcel, tool_name: str, started_at: datetime) -> Path:
    return output_root / parcel.name / tool_name / started_at.astimezone().strftime("%Y-%m-%d_%H%M%S")


def latest_result(output_root: Path, parcel: Parcel, tool_name: str) -> dict[str, Any] | None:
    tool_dir = output_root / parcel.name / tool_name
    if not tool_dir.is_dir():
        return None
    for folder in sorted(tool_dir.iterdir(), reverse=True):
        result_file = folder / RESULT_FILE
        if not result_file.is_file():
            continue
        data: dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
        if data["run"]["status"] != RunStatus.ERROR:
            return data
    return None
