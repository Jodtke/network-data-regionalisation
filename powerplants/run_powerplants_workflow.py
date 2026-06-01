from __future__ import annotations

"""Run plant-related regionalisation modules in a reproducible order.

Thermal units, other non-RES, other RES, BESS, and availability profiles share
intermediate allocation bases. This runner keeps the execution order explicit so
that residual capacities and fallback bases are generated before downstream
modules consume them.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from powerplants_common import load_yaml_config, resolve_path

LOG = logging.getLogger(__name__)

SCRIPT_BY_STEP = {
    "thermal": "thermal_disaggregation.py",
    "other_nonres": "other_nonres_disaggregation.py",
    "other_res": "other_res_disaggregation.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the powerplants preprocessing workflow.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        choices=sorted(SCRIPT_BY_STEP),
        help="Subset of steps to run. Defaults to workflow.steps in YAML or all steps.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_path = resolve_path(args.config, Path.cwd())
    assert config_path is not None
    cfg: dict[str, Any] = load_yaml_config(config_path)
    steps = args.steps or cfg.get("steps") or ["thermal", "other_nonres", "other_res"]
    script_dir = Path(__file__).resolve().parent
    for step in steps:
        script_name = SCRIPT_BY_STEP[str(step)]
        cmd = [sys.executable, str(script_dir / script_name), "--config", str(config_path)]
        LOG.info("running %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
