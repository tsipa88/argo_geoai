# Test parcels

Field outlines from the open Fields of The World dataset (Source Cooperative, CC BY 4.0), Moldova
partition, year 2025, picked for what they show rather than for who farms them. Holes left by the raster-to-vector step
were dropped and the outlines simplified to 2 m. They are model
predictions from Sentinel-2 imagery, not cadastral parcels, and are not connected to any client.

| File | FTW id | What it shows |
|---|---|---|
| `field_slope.geojson` | 41665 | one sloping field with a single late crop (mean slope about 3°, east aspect, no inflow) |
| `field_ravines.geojson` | 301853 | steep dissected slopes crossed by runoff from a catchment larger than the field |
| `field_two_crops.geojson` | 367534 + 367387 | two neighbouring flat fields with different crops under one polygon |
| `field_two_parts.geojson` | 367399 + 367404 | two separate flat fields with early (winter) crops as one MultiPolygon |

The integration tests in `tests/integration/` (marker `network`) assert the numbers these fields give
for the 2026 season; the unit tests only need the geometries.
