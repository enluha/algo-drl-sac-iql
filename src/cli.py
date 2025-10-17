from __future__ import annotations

import argparse

from src.run_offline_pretrain import main as offline_main
from src.run_sac_finetune import main as sac_main
from src.run_walkforward import main as walk_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="algo-drl-sac-iql CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--config", default="config/config.yaml")
        sub.add_argument("--device", default=None)
        sub.add_argument("--seed", type=int, default=None)
        sub.add_argument("--n-workers", type=int, default=None)
        sub.add_argument("--log-level", default=None)
        sub.add_argument("--blas-threads", type=int, default=None)
        return sub

    add_common(subparsers.add_parser("offline-pretrain", help="Run offline IQL pretraining"))
    add_common(subparsers.add_parser("sac-finetune", help="Run SAC fine-tuning"))
    add_common(subparsers.add_parser("walkforward", help="Run walk-forward evaluation"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_args = ["--config", args.config]
    if args.device:
        base_args.extend(["--device", args.device])
    if args.seed is not None:
        base_args.extend(["--seed", str(args.seed)])
    if args.n_workers is not None:
        base_args.extend(["--n-workers", str(args.n_workers)])
    if args.log_level:
        base_args.extend(["--log-level", args.log_level])
    if args.blas_threads is not None:
        base_args.extend(["--blas-threads", str(args.blas_threads)])

    if args.command == "offline-pretrain":
        offline_main(base_args)
    elif args.command == "sac-finetune":
        sac_main(base_args)
    elif args.command == "walkforward":
        walk_main(base_args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
