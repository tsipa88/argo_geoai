from pathlib import Path

import pytest

from argo_geoai.config import get_settings
from argo_geoai.parcels import Parcel, from_request
from argo_geoai.request import ParcelRequest

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def open_service(monkeypatch):
    monkeypatch.setenv("ARGO_GEOAI_API_TOKEN", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def load_parcel(path: Path) -> Parcel:
    return from_request(ParcelRequest.from_json(path.read_text(encoding="utf-8"), default_name=path.stem))


@pytest.fixture
def field_slope_geojson() -> Path:
    return DATA / "field_slope.geojson"


@pytest.fixture
def field_slope(field_slope_geojson: Path) -> Parcel:
    return load_parcel(field_slope_geojson)
