"""Dataset splitting and persistence logic."""
from __future__ import annotations

import json
import random
from pathlib import Path


def find_existing_split(dataset_name: str) -> Path | None:
    """Most recent persisted split file for a dataset name, or None.

    Filenames look like ``<name>__seed<seed>__v<val>__t<test>.json``; multiple can
    exist if a dataset was trained with different ratios/seeds — pick the newest.
    """
    split_dir = Path("splits")
    if not split_dir.is_dir():
        return None
    matches = sorted(split_dir.glob(f"{dataset_name}__*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_or_create_split(
    samples: list[tuple[Path, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    data_dir: str | Path,
    split_file_override: str | None = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]], str]:
    """
    Returns (train, val, test, split_hash) and persists the split assignment to ensure
    future runs and evaluations use the exact same test set.
    """
    data_dir = Path(data_dir)
    # The basename of the dataset dir is used as the dataset name in the split filename
    dataset_name = data_dir.name
    
    # We map absolute sample paths to their relative names for stable storage
    sample_names = sorted([p.name for p, _ in samples])
    name_to_sample = {p.name: (p, label) for p, label in samples}

    if split_file_override:
        split_path = Path(split_file_override)
    else:
        split_dir = Path("splits")
        split_dir.mkdir(exist_ok=True)
        split_name = f"{dataset_name}__seed{seed}__v{val_ratio}__t{test_ratio}.json"
        split_path = split_dir / split_name

    if split_path.exists():
        print(f"[splits] Loading existing split from {split_path}")
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        
        train_names = set(split_data.get("train", []))
        val_names = set(split_data.get("val", []))
        test_names = set(split_data.get("test", []))
        
        train = [name_to_sample[n] for n in sample_names if n in train_names and n in name_to_sample]
        val = [name_to_sample[n] for n in sample_names if n in val_names and n in name_to_sample]
        test = [name_to_sample[n] for n in sample_names if n in test_names and n in name_to_sample]
        
        missing = len(sample_names) - (len(train) + len(val) + len(test))
        if missing > 0:
            print(f"[splits] Warning: {missing} files in current dataset were not found in the loaded split file. They will be ignored.")
        
        return train, val, test, split_path.stem

    # Create a new split
    print(f"[splits] Creating new split -> {split_path}")
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    
    n_val = int(len(samples) * val_ratio)
    n_test = int(len(samples) * test_ratio)
    
    # Ensure at least 1 sample in val and test if possible, unless ratio is 0
    if n_val == 0 and val_ratio > 0 and len(samples) > 2:
        n_val = 1
    if n_test == 0 and test_ratio > 0 and len(samples) > 2:
        n_test = 1
        
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test : n_test + n_val])
    
    train, val, test = [], [], []
    train_names, val_names, test_names = [], [], []
    
    for i, s in enumerate(samples):
        if i in test_idx:
            test.append(s)
            test_names.append(s[0].name)
        elif i in val_idx:
            val.append(s)
            val_names.append(s[0].name)
        else:
            train.append(s)
            train_names.append(s[0].name)
            
    split_data = {
        "dataset": dataset_name,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train": train_names,
        "val": val_names,
        "test": test_names
    }
    
    split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    return train, val, test, split_path.stem
