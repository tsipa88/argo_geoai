from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Esri World Imagery: the tiles are Esri's, used under Esri's Terms of Use; see README "Data sources"
ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_WORLD_IMAGERY_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics, and the GIS User Community"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARGO_GEOAI_", env_file=(PROJECT_ROOT / ".env", ".env"), extra="ignore"
    )

    output_dir: Path = Path("output")
    stac_url: str = PLANETARY_COMPUTER_STAC
    api_token: str | None = None
    max_parallel_runs: int = 2
    cache_dir: Path = PROJECT_ROOT / "cache"
    # XYZ tile URL of the satellite basemap under the field maps; empty disables the basemap
    imagery_tiles: str = ESRI_WORLD_IMAGERY
    imagery_attribution: str = ESRI_WORLD_IMAGERY_ATTRIBUTION


@lru_cache
def get_settings() -> Settings:
    return Settings()
