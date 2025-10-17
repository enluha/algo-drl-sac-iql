from pathlib import Path
import yaml, json, tempfile, shutil

def load_yaml(path: str | Path):
    p = Path(path)
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "include" in data:
        merged = {}
        for rel in data["include"]:
            sub = p.parent / rel
            merged[sub.stem] = load_yaml(sub)
        return merged
    return data

def save_json(obj, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv(df, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=True)

def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), str(p))

def save_html(fig, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p))
