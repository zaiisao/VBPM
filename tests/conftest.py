def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running stability tests; run with `pytest -m slow`",
    )
