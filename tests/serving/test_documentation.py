import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


def test_documented_requests_match_openapi_and_execute(settings: ServingSettings) -> None:
    guide = (Path(__file__).resolve().parents[2] / "docs" / "serving.md").read_text(
        encoding="utf-8"
    )
    with TestClient(create_app(settings)) as client:
        for marker, endpoint, expected in [
            ("single-request", "/v1/score", APPLICANT_EXAMPLE),
            ("batch-request", "/v1/score/batch", {"applicants": [APPLICANT_EXAMPLE]}),
        ]:
            match = re.search(r"<!-- " + marker + r" -->\s*```json\s*(.*?)\s*```", guide, re.DOTALL)
            assert match is not None
            payload = json.loads(match.group(1))
            assert payload == expected
            assert client.post(endpoint, json=payload).status_code == 200
