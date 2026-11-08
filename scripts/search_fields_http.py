"""Development helper: POST /v1/runs with tool search_fields, in process, without serving it publicly."""

import argparse
import json
import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from argo_geoai import server
from argo_geoai.config import get_settings
from argo_geoai.request import ParcelRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--geometry", required=True, help="GeoJSON Feature file, or - for stdin")
    parser.add_argument("--params", default="{}", help="tool params as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.geometry == "-":
        request = ParcelRequest.from_json(sys.stdin.read())
    else:
        path = Path(args.geometry)
        request = ParcelRequest.from_json(path.read_text(encoding="utf-8-sig"), default_name=path.stem)

    body = {
        "tool": "search_fields",
        "geometry": request.model_dump(mode="json", exclude_none=True),
        "params": json.loads(args.params),
    }
    token = get_settings().api_token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = TestClient(server.app).post("/v1/runs", json=body, headers=headers)
    folder = get_settings().output_dir / request.name / "search_fields_http"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "response.json").write_text(json.dumps(response.json(), ensure_ascii=False, indent=2))
    if response.status_code != 200:
        summary = {"status_code": response.status_code, "response": str(folder / "response.json")}
        sys.stdout.write(json.dumps(summary) + "\n")
        raise SystemExit(1)
    fields = response.json()["result"]["metrics"]["fields"]
    (folder / "fields.geojson").write_text(json.dumps(fields, ensure_ascii=False))
    sys.stdout.write(
        json.dumps(
            {
                "status_code": response.status_code,
                "status": response.json()["status"],
                "response": str((folder / "response.json").resolve()),
                "fields": str((folder / "fields.geojson").resolve()),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
