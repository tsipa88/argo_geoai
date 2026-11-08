from typing import Any

from argo_geoai.tools.activity import ActivityTool
from argo_geoai.tools.base import Tool
from argo_geoai.tools.field_details import FieldDetailsTool
from argo_geoai.tools.moisture import MoistureTool
from argo_geoai.tools.relief import ReliefTool
from argo_geoai.tools.relief_v2 import ReliefV2Tool
from argo_geoai.tools.search_fields import SearchFieldsTool
from argo_geoai.tools.vegetation import VegetationTool
from argo_geoai.tools.vegetation_v2 import VegetationV2Tool
from argo_geoai.tools.water import WaterTool
from argo_geoai.tools.water_v2 import WaterV2Tool

TOOLS: dict[str, Tool[Any, Any]] = {
    tool.name: tool
    for tool in (
        VegetationTool(),
        ReliefTool(),
        WaterTool(),
        VegetationV2Tool(),
        ReliefV2Tool(),
        WaterV2Tool(),
        MoistureTool(),
        ActivityTool(),
        SearchFieldsTool(),
        FieldDetailsTool(),
    )
}

__all__ = ["TOOLS", "Tool"]
