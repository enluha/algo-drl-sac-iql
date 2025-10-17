from __future__ import annotations
import os, torch, logging

def get_torch_device(prefer: str | None = None) -> torch.device:
    forced = os.getenv("QA_DEVICE", None)
    want = (forced or prefer or "").lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_num_threads(n: int) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))

def log_device(logger: logging.Logger) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    msg = f"Device: {dev}"
    if dev == "cuda":
        props = torch.cuda.get_device_properties(0)
        msg += f" | {props.name} cc={props.major}.{props.minor} total_mem={props.total_memory/1e9:.1f}GB"
    logger.info(msg)
