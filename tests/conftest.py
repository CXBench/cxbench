from __future__ import annotations

import os


def pytest_runtest_setup(item):
    os.environ["CXBENCH_ALLOW_TEST_DETERMINISTIC_USER_SIM"] = "1"
