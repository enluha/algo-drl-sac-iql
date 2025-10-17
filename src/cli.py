import argparse, os
from src.utils.seed import seed_everything

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("offline-pretrain","sac-finetune","walkforward"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="config/config.yaml")
        sp.add_argument("--device", choices=["cpu","cuda"], default=None)
        sp.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CONFIG"] = args.config
    if args.device: os.environ["QA_DEVICE"]=args.device

    seed_everything(args.seed)

    if args.cmd=="offline-pretrain":
        from src.drl.offline.iql_pretrain import main as run; run()
    elif args.cmd=="sac-finetune":
        from src.drl.online.sac_train import main as run; run()
    else:
        from src.run_walkforward import run; run()

if __name__ == "__main__":
    main()
