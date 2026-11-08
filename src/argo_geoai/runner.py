import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from argo_geoai import __version__
from argo_geoai.parcels import Parcel
from argo_geoai.results import RESULT_FILE, run_folder
from argo_geoai.schema import Message, RunInfo, RunStatus, ToolInfo, ToolResult
from argo_geoai.tools.base import RunContext, Tool, ToolOutput

log = logging.getLogger(__name__)


def run_tool[P: BaseModel, M: BaseModel](
    tool: Tool[P, M],
    parcel: Parcel,
    params: P,
    output_root: Path,
    *,
    vegetation_result: dict[str, Any] | None = None,
) -> tuple[ToolResult[M, P], Path]:
    started_at = datetime.now(UTC)
    clock = time.perf_counter()
    workdir = run_folder(output_root, parcel, tool.name, started_at)
    workdir.mkdir(parents=True, exist_ok=False)

    output: ToolOutput[M] | None = None
    errors: list[Message] = []
    try:
        output = tool.run(
            params,
            RunContext(
                parcel=parcel,
                workdir=workdir,
                output_root=output_root,
                vegetation_result=vegetation_result,
            ),
        )
    except Exception as error:
        log.exception("%s failed for %s", tool.name, parcel.name)
        errors.append(Message(code=type(error).__name__, text=str(error)))

    if output is None:
        status = RunStatus.ERROR
    elif output.partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.OK

    result: ToolResult[M, P] = ToolResult(
        tool=ToolInfo(name=tool.name, version=tool.version, title=tool.title),
        run=RunInfo(
            id=uuid.uuid4(),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_s=round(time.perf_counter() - clock, 2),
            status=status,
            status_label=status.label,
            argo_geoai_version=__version__,
        ),
        parcel=parcel.info(),
        parameters=params,
        data_sources=output.data_sources if output else [],
        summary=output.summary if output else [],
        metrics=output.metrics if output else None,
        artifacts=output.artifacts if output else [],
        warnings=parcel.checks() + (output.warnings if output else []),
        errors=errors,
    )
    result.write(workdir / RESULT_FILE)
    return result, workdir
