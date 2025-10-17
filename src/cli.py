from __future__ import annotations

import argparse
from importlib import import_module


COMMAND_MODULES = {
    "offline-pretrain": "src.run_offline_pretrain",
    "sac-finetune": "src.drl.online.sac_train",
    "walkforward": "src.run_walkforward",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="algo-drl-sac-iql CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default="config/config.yaml")
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--seed", type=int, default=None)
        subparser.add_argument("--n-workers", type=int, default=None)
        subparser.add_argument("--log-level", default=None)

    add_common(subparsers.add_parser("offline-pretrain"))
    add_common(subparsers.add_parser("sac-finetune"))
    add_common(subparsers.add_parser("walkforward"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module_name = COMMAND_MODULES[args.command]
    module = import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
