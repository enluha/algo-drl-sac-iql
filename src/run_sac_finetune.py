import argparse, os
from src.utils.seed import seed_everything
from src.drl.online.sac_train import main as run

def main():
    run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CONFIG"] = args.config
    if args.device: os.environ["QA_DEVICE"] = args.device
    seed_everything(args.seed)
    run()
