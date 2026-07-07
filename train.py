"""Train a captcha OCR model.

    python train.py --config configs/data_2.yaml
    python train.py --data-dir data_1 --out-dir runs/data_1 --epochs 300
"""
import argparse

from ocr.config import Config
from ocr.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--data-dir", dest="data_dir")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--num-workers", dest="num_workers", type=int)
    ap.add_argument("--early-stop-patience", dest="early_stop_patience", type=int)
    args = vars(ap.parse_args())
    cfg = Config.load(args.pop("config"), overrides=args)
    train(cfg)


if __name__ == "__main__":
    main()
