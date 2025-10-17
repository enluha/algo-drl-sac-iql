from __future__ import annotations

import argparse

from src.drl.online import sac_train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC fine-tuning")
    parser.add_argument("--config", default="config/config.yaml", help="Path to master config")
    parser.add_argument("--device", default=None, help="Preferred torch device")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blas-threads", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {
        "seed": args.seed,
        "blas_threads": args.blas_threads,
        "log_level": args.log_level,
        "n_workers": args.n_workers,
    }
    sac_train.run(args.config, prefer_device=args.device, overrides=overrides)


if __name__ == "__main__":
    main()
