from __future__ import annotations

"""Run the DSR extraction and regionalisation steps for a scenario."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DSR preprocessing workflow.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-timeseries", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_path = args.config.resolve()
    script_dir = Path(__file__).resolve().parent
    cmd = [sys.executable, str(script_dir / "dsr_disaggregation.py"), "--config", str(config_path)]
    if args.skip_timeseries:
        cmd.append("--skip-timeseries")
    LOG.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
