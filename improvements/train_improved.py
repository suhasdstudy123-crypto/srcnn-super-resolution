"""Train the improved models (written by us).

Everything except the model and the loss is the SAME as the Stanford replication run `main`:
same 800 train images, same 50 val / 50 test split, Adam lr 1e-3, batch 2, 100 epochs,
keep the best-on-val weights, test once at the end. So any difference in the results comes
from the improvement alone.

Reuses from src/ (unchanged): get_datasets, train_one_epoch, evaluate, set_seed.
Saves to <out_dir>/<run_name>/ the same files as src/train.py:
    config.json, history.csv, best.pth, last.pth (for resuming), test.csv, test_per_image.csv

The three improvement runs (each adds one thing to the previous):
    --model residual                                    --run_name imp1_residual
    --model deep10                                      --run_name imp2_deep10
    --model deep10 --loss mse_ssim --ssim_weight W      --run_name imp3_deep10_ssim

SSIM-weight sweep (choose W on VAL only, the test set is not touched):
    --model deep10 --loss mse_ssim --ssim_weight 0.05 --epochs 10 --no_test --run_name sweep_w0.05
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import get_datasets  # noqa: E402
from model import count_params  # noqa: E402
from train import evaluate, train_one_epoch  # noqa: E402  (the replication loop, unchanged)
from utils import set_seed  # noqa: E402

from losses import make_loss  # noqa: E402
from models import MODELS, build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True, help="folder with train_224.npy and valid_224.npy")
    p.add_argument("--out_dir", required=True, help="e.g. <drive>/improvement-checkpoints")
    p.add_argument("--run_name", required=True)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--loss", default="mse", choices=["mse", "mse_ssim"])
    p.add_argument("--ssim_weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_train", type=int, default=None, help="use only the first N train images")
    p.add_argument("--no_test", action="store_true", help="skip the test set (for the weight sweep)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def run_test(model, run_dir, test_loader, best_psnr, run_name, device):
    """Load the best-on-val weights and score the 50 test images ONCE."""
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    psnr, ssim, mse = evaluate(model, test_loader, device)
    test = {"run": run_name, "test_psnr": psnr.mean().item(), "test_ssim": ssim.mean().item(),
            "test_mse": mse.mean().item(), "best_val_psnr": best_psnr}
    pd.DataFrame([test]).round(4).to_csv(run_dir / "test.csv", index=False)
    pd.DataFrame({"psnr": psnr.numpy(), "mse": mse.numpy()}).to_csv(
        run_dir / "test_per_image.csv", index_label="image")
    print("TEST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # ----- data: identical to the replication -----
    train_ds, val_ds, test_ds = get_datasets(None, None, args.cache_dir, n_train=args.n_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=10, num_workers=args.num_workers)

    # ----- the only things that change: model and loss -----
    model = build_model(args.model).to(device)
    loss_fn = make_loss(args.loss, args.ssim_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_text = args.loss + (f" (ssim_weight {args.ssim_weight})" if args.loss == "mse_ssim" else "")
    print(f"Device {device} | model {args.model} ({count_params(model):,} params) | loss {loss_text} | "
          f"train {len(train_ds)} val {len(val_ds)} test {len(test_ds)} | run dir {run_dir}")

    # ----- resume if Colab cut a previous run of the same name -----
    start_epoch, best_psnr, history = 1, -1.0, []
    last_path = run_dir / "last.pth"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_psnr, history = ckpt["epoch"] + 1, ckpt["best_psnr"], ckpt["history"]
        print(f"Resuming from epoch {start_epoch} (best val PSNR so far {best_psnr:.3f})")

    # ----- training loop (same as src/train.py) -----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_psnr, val_ssim, val_mse = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_mse.mean().item(),
               "val_psnr": val_psnr.mean().item(), "val_ssim": val_ssim.mean().item(),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if row["val_psnr"] > best_psnr:                 # best epoch judged on VAL only
            best_psnr = row["val_psnr"]
            torch.save(model.state_dict(), run_dir / "best.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_psnr": best_psnr, "history": history}, last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch {epoch:3d} | train loss {train_loss:.5f} | val PSNR {row['val_psnr']:.3f} "
              f"SSIM {row['val_ssim']:.4f} | best {best_psnr:.3f} | {row['seconds']}s", flush=True)

    if args.no_test:
        print("Skipped the test set (--no_test). Compare runs on val in history.csv.")
    else:
        run_test(model, run_dir, test_loader, best_psnr, args.run_name, device)


if __name__ == "__main__":
    main()
