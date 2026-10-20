"""
Entrypoint: python -m gateway
"""
from __future__ import annotations

import sys


def main() -> None:
    if "--test" in sys.argv:
        print(
            "The hand-rolled --test runner is gone; it duplicated the pytest "
            "suite against the deleted _legacy module. Use:\n"
            "  python -m pytest -q                 every test\n"
            '  python -m pytest -q -m "not slow"   fast edit-test loop'
        )
        raise SystemExit(2)

    import os
    from pathlib import Path

    import uvicorn

    from gateway.app import GATEWAY_VERSION, app
    from gateway.config import GatewayConfig

    # Honor an explicit settings.json path from the GUI subprocess launcher.
    cfg_path = os.environ.get("LOCAL_LLM_GATEWAY_CONFIG_PATH")
    config = GatewayConfig.from_env(
        config_path=Path(cfg_path) if cfg_path else None)
    # GATEWAY_VERSION, never a literal -- this is the line a headless start
    # prints, and it said v6.1.0 for two releases. See
    # `test_no_version_literals_outside_the_one_definition`.
    print(f"faststrata-gateway v{GATEWAY_VERSION} listening on "
          f"{config.host}:{config.port} -> {config.upstream}")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
