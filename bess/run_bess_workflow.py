from __future__ import annotations

"""Run the BESS disaggregation step for a configured scenario."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BESS preprocessing workflow.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_path = args.config.resolve()
    script_dir = Path(__file__).resolve().parent
    cmd = [sys.executable, str(script_dir / "bess_disaggregation.py"), "--config", str(config_path)]
    LOG.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
