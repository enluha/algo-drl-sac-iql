"""
MVP - CLI orchestrator: sets BLAS threads, passes n_workers/log_level through.
"""

import os, argparse

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/config.yaml")
    common.add_argument("--blas-threads", type=int, default=None)
    common.add_argument("--n-workers", type=int, default=None)
    common.add_argument("--log-level", default=None)

    sub.add_parser("train", parents=[common])
    sub.add_parser("backtest", parents=[common])
    sub.add_parser("report", parents=[common])
    return p.parse_args()

def set_env_threads(k: int | None):
    if k is None: return
    os.environ["OMP_NUM_THREADS"] = str(k)
    os.environ["MKL_NUM_THREADS"] = str(k)
    os.environ["OPENBLAS_NUM_THREADS"] = str(k)
    os.environ["NUMEXPR_NUM_THREADS"] = str(k)

def main():
    args = parse_args()
    # load runtime to get defaults for blas/n_workers/log_level
    from .utils.io_utils import load_yaml
    runtime = load_yaml("config/runtime.yaml")
    blas_threads = args.blas_threads or runtime.get("blas_threads", 6)
    set_env_threads(blas_threads)

    # Inject effective runtime overrides into args for downstream
    args.n_workers = args.n_workers if args.n_workers is not None else runtime.get("n_workers", 1)
    args.log_level = args.log_level if args.log_level is not None else runtime.get("log_level", "INFO")
    args.blas_threads = blas_threads

    from . import main_train, main_backtest, main_report
    if args.cmd == "train":     main_train.main(args)
    elif args.cmd == "backtest": main_backtest.main(args)
    elif args.cmd == "report":   main_report.main(args)


if __name__ == "__main__":
    main()
