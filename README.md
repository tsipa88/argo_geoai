# argo_geoai

Satellite analysis of land parcels: send a polygon, get numbers, conclusions and interactive Plotly maps.
Built on [GeoAI](https://github.com/opengeos/geoai) and Microsoft Planetary Computer (free, no API key).

The service only takes a polygon as a GeoJSON Feature and keeps no state. It works as a command line tool and
as an HTTP server.

## Tools

| Tool | What it does | Data |
|---|---|---|
| `vegetation` | NDVI history over a period: per-date statistics, peak, data gaps, zones of the field with different NDVI curves (several crops under one polygon), land cover shares | Sentinel-2 L2A, ESA WorldCover 2021 |
| `relief` | Elevation, slope with erosion classes, aspect, the low part of the field, NDVI against relief | Copernicus DEM GLO-30, Sentinel-2 |
| `water` | Where water collects (topographic wetness index), concentrated runoff, inflow from outside the field, open water seen since 1984, NDVI against wetness | Copernicus DEM GLO-30, JRC Global Surface Water, Sentinel-2 |
| `moisture` | Water stress of plants: NDMI over the season on a closed canopy, parts of the field drier than their NDVI explains | Sentinel-2 L2A |
| `activity` | What was done on the field and when: harvest, tillage, sowing, emergence, green cut, as dated events and a date / activity table. Optionally reports what happened on a smaller area inside the polygon (`properties.highlight`) | Sentinel-2 L2A, Sentinel-1 RTC |
| `search_fields` | Fields that meet the polygon, with their areas and overlaps, as GeoJSON with straightened outlines | Fields of The World (CC-BY-4.0) |
| `field_details` | A report on one built or commercial parcel: land cover, relief, dated changes of buildings and surfaces, objects lying across the boundary, true-colour snapshots per year | Sentinel-2 L2A, Sentinel-1 RTC, ESA WorldCover 2021, Copernicus DEM GLO-30 |

Parameters and results of every tool are described by JSON Schemas in `schemas/`.

## Examples rendering with Plotly

<img width="1049" height="534" alt="1" src="https://github.com/user-attachments/assets/8e8a31d4-c8e1-4ede-acee-92cdb44f1e19" />
<img width="1057" height="322" alt="2" src="https://github.com/user-attachments/assets/29c69be4-43ca-42a0-a21b-0161efa214cc" />
<img width="1057" height="498" alt="3" src="https://github.com/user-attachments/assets/bbce9550-6544-473d-8355-118c5054bdc3" />
<img width="1054" height="490" alt="4" src="https://github.com/user-attachments/assets/7136c186-e8f9-4b6e-8028-2c3b850dd1ea" />
<img width="1052" height="658" alt="5" src="https://github.com/user-attachments/assets/becaf5eb-ac0b-421c-9c33-cc58f2850251" />
<img width="1053" height="536" alt="6" src="https://github.com/user-attachments/assets/70875f5f-2def-4690-9c55-470f3394c927" />
<img width="1053" height="314" alt="7" src="https://github.com/user-attachments/assets/d3bc0191-72e1-45f4-b0aa-f5a292325c65" />
<img width="1056" height="493" alt="8" src="https://github.com/user-attachments/assets/3e9880da-8473-4fbd-a51a-9f4143c95656" />
<img width="1057" height="375" alt="9" src="https://github.com/user-attachments/assets/4ec9eba0-4594-47b5-bcd9-5e7394d53048" />


## Running

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run argo-geoai tools
uv run argo-geoai run-all --geometry parcel.geojson
uv run argo-geoai run moisture --geometry parcel.geojson
uv run argo-geoai run relief_v2 --geometry - < parcel.geojson
```

`--geometry -` reads stdin. In the command line `vegetation`, `relief` and `water` write PNG images and their
`*_v2` versions write interactive HTML; `run-all` and the HTTP server use the `*_v2` versions.

Input is a GeoJSON Feature in lng/lat (`schemas/request.schema.json`). A FeatureCollection with one Feature or
a bare geometry works too:

```json
{
  "type": "Feature",
  "geometry": {"type": "Polygon", "coordinates": [[[28.65, 45.95], [28.66, 45.95], [28.66, 45.96], [28.65, 45.95]]]},
  "properties": {"name": "my_field", "area_ha": 43.0, "metadata": {"any": "caller data"}}
}
```

All properties are optional: `name` names the output folder, `area_ha` enables the area check, `metadata` comes
back untouched.

Results go to `output/<name>/<tool>/<YYYY-MM-DD_HHMMSS>/`, and stdout gets one JSON line with statuses and paths.
Exit code: `0` when no run failed, `1` when a run has status `error`, `2` for invalid input.

## HTTP API

```bash
uv run argo-geoai serve --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /health` | Status and version, always open |
| `GET /v1/tools` | Tool types and the implementation behind each |
| `POST /v1/runs` | Runs one tool synchronously and returns its result and figures |
| `GET /docs` | OpenAPI pages, only when no token is set |

With `ARGO_GEOAI_API_TOKEN` set, `/v1/*` needs `Authorization: Bearer <token>`.

```bash
curl -s http://127.0.0.1:8000/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"tool": "vegetation", "geometry": {"type": "Feature", "geometry": {...}, "properties": {}}, "params": {}}'
```

Response:

```json
{
  "tool": "vegetation",
  "tool_version": "vegetation_v2/1.0.0",
  "status": "ok",
  "result": {},
  "figures": [{"name": "maps", "title": "...", "figure": {}}]
}
```

`result` has the same envelope for every tool (`summary`, `metrics`, `warnings`, `errors`, ...), `figure` is a
Plotly spec for `Plotly.newPlot(el, figure.data, figure.layout)`. A request with an unknown tool, a broken polygon
or bad params gets 422.

`relief`, `water` and `moisture` take the latest vegetation result in `context.vegetation` of the request.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ARGO_GEOAI_API_TOKEN` | empty | Bearer token for `/v1/*`; empty means no check |
| `ARGO_GEOAI_MAX_PARALLEL_RUNS` | `2` | Analyses at once, further requests wait |
| `ARGO_GEOAI_OUTPUT_DIR` | `output` | Root of the run folders |
| `ARGO_GEOAI_IMAGERY_TILES` | Esri World Imagery | XYZ tile URL of the satellite basemap under the `search_fields` and `field_details` maps; empty disables the basemap |
| `ARGO_GEOAI_IMAGERY_ATTRIBUTION` | Esri's credit line | Attribution shown on the maps and pages for that basemap |

## Deployment

Ubuntu x86_64 with nginx and Passenger >= 6.0. Passenger starts `argo-geoai serve` on a local port and proxies
the domain to it. Python and all packages come from `uv`, no system GDAL is needed.

Requirements:

* Memory: about 0.3 GB idle and 0.25 GB per running analysis on a 200 ha field.
* Disk: about 2.5 GB for `.venv`, 2 GB for the `search_fields` map.
* Outbound HTTPS to `planetarycomputer.microsoft.com`, `*.blob.core.windows.net` and `data.source.coop`.
* An analysis takes minutes, so the domain must not sit behind a proxy that cuts long requests (Cloudflare: DNS
  only). Passenger open source allows up to 10 minutes per request.

First install, as the user that will run the app (`APP_ROOT` below is the folder of the checkout):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository url> APP_ROOT
cd APP_ROOT
cp .env.example .env && chmod 600 .env
~/.local/bin/uv sync --frozen --no-dev
.venv/bin/argo-geoai tools
```

Put a token into `.env` (`openssl rand -hex 32`). Then copy `deploy/nginx.conf` to
`/etc/nginx/sites-available/argo_geoai`, replace `geoai.example.com`, `APP_ROOT` and `APP_USER` in it, link it
into `sites-enabled` and run:

```bash
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d geoai.example.com
curl https://geoai.example.com/health
```

Download the `search_fields` map once, or the first request will wait for it:

```bash
.venv/bin/python -c "from argo_geoai.sources import field_boundaries as f; print(f.country_file(f.PARTITIONS[0]))"
```

Every deploy, with the checkout folder as the argument (the script checks that it is an argo_geoai checkout)
and optionally a branch (`main` by default):

```bash
HEALTH_URL=https://geoai.example.com/health APP_ROOT/deploy/deploy.sh APP_ROOT
```

The app's output is in `/var/log/nginx/error.log`, requests are in `/var/log/nginx/argo_geoai.access.log`.

## Data sources, licences and attribution

The code is under the [MIT licence](LICENSE). The data it reads is not: each source has its own terms, and
whoever runs the service and shows its results is responsible for meeting them. The HTML pages and the
`data_sources` field of every result carry the notices below.

| Source | Terms | Required credit |
|---|---|---|
| Sentinel-1 RTC, Sentinel-2 L2A | [Copernicus Sentinel data legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice): free use, modification and redistribution | "Contains modified Copernicus Sentinel data [year]" |
| Copernicus DEM GLO-30 | [Copernicus DEM licence](https://spacedata.copernicus.eu/documents/20123/121286/CSCDA_ESA_Mission-specific+Annex_31_Oct_22.pdf): free use, the credit line is mandatory | "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" |
| ESA WorldCover 2021 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium" |
| JRC Global Surface Water | [Free, with citation](https://global-surface-water.appspot.com/download) | Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S. (2016). High-resolution mapping of global surface water and its long-term changes. Nature 540, 418-422 |
| Fields of The World field boundaries | [CC BY 4.0](https://source.coop/repositories/fiboa/ftw-global-data) | "Fields of The World (Source Cooperative), CC BY 4.0" |
| Microsoft Planetary Computer | [Terms of use](https://planetarycomputer.microsoft.com/terms): data access, no key needed, no uptime promise | - |
| Esri World Imagery (map basemap) | [Esri Terms of Use](https://www.esri.com/en-us/legal/terms/full-master-agreement) | "Esri, Maxar, Earthstar Geographics, and the GIS User Community" |

**Esri World Imagery.** The maps of `search_fields` and `field_details` are drawn over Esri's World Imagery
tiles. Esri lets its basemaps be used in third-party apps, but ArcGIS Online basemap access in a
production app is meant to go through an [ArcGIS Location Platform](https://location.arcgis.com/) account
(there is a free tier), and the tiles may not be stored, redistributed or used offline. Before a public or
commercial deployment either register an account and point `ARGO_GEOAI_IMAGERY_TILES` at a basemap URL
you are entitled to use, or switch the basemap off with an empty `ARGO_GEOAI_IMAGERY_TILES`. The results
themselves do not depend on it.

**Fields of The World.** The `search_fields` map is a machine-learning prediction of field outlines from
Sentinel-2 imagery. It is not a cadastre and says nothing about who owns or farms a field.

### Disclaimer

The results are automated estimates from open satellite data at 10-30 m resolution, for information
only. They are not a cadastral, legal, agronomic or insurance document, do not replace a survey on the
ground, and come with no warranty of accuracy or availability (see the licence). Polygons sent to the
service are the caller's responsibility: do not send data you are not entitled to process, and keep in
mind that a parcel outline plus a public cadastral map can identify a landowner.

The polygons in `tests/data/` are field outlines from the Fields of The World dataset in Moldova, picked
for their relief and crop pattern, not parcels of any client.

## Development

```bash
uv run ruff format && uv run ruff check
uv run mypy
uv run pytest
uv run pytest -m network tests/integration
```

The integration tests download a season of satellite data for the four parcels in `tests/data/` and take
about 20 minutes; they are skipped by default.
