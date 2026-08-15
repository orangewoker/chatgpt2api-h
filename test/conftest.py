from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that require a local server, configured accounts, and upstream network",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="live test: pass --run-live to enable")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
