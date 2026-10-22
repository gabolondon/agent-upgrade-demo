import os

os.environ["LLM_FAKE"] = "1"
os.environ["FAKE_LATENCY_SCALE"] = "0"

import httpx  # noqa: E402
import pytest  # noqa: E402

from agent_common.tools import ApiClient  # noqa: E402
from common.config import Settings  # noqa: E402
from portfolio_api.app import create_app  # noqa: E402


def make_api(fail_rate: float = 1.0) -> ApiClient:
    app = create_app(overhead_ms=0, ms_per_1k_rows=0, fail_rate=fail_rate)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    return ApiClient("http://api", http)


@pytest.fixture
def api():
    return make_api()


@pytest.fixture
def settings():
    return Settings.from_env()
