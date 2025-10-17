from src.utils.io_utils import atomic_write_text

def build_text_report(summary: dict, path, context: dict):
    lines = []
    lines += ["Strategy Overview:", f"  {context.get('overview','DRL IQL→SAC walk-forward')}", ""]
    lines += ["Parameters:"] + [f"  {k}: {v}" for k,v in context.get("params",{}).items()] + [""]
    lines += ["Equity & Risk:"] + [f"  {k}: {v}" for k,v in summary.items()]
    atomic_write_text(path, "\n".join(lines))
