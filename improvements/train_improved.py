"""Train the improved models. Same data, same split, same training loop, same settings as
Stanford's run (Adam, lr 1e-3, batch 2, 100 epochs, best-on-val, test once) so every
difference in the results comes from the improvement alone.

Reuses src/: the dataset, the per-epoch training/evaluation functions and the metrics.
Saves to <out_dir>/<run_name>/ exactly like src/train.py (history.csv, best.pth, last.pth,
test.csv, test_per_image.csv, config.json), and resumes if a run was cut off.

Runs (each adds one thing to the previous):
  1  --arch srcnn_res                          --run_name imp1_residual
  2  --arch deep --n_layers 10                 --run_name imp2_deep10
  3  --arch deep --n_layers 10 --loss mse_ssim --run_name imp3_deep10_ssim
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
from train import evaluate, train_one_epoch  # noqa: E402  (the original loop, unchanged)
from utils import set_seed  # noqa: E402

from losses import make_loss  # noqa: E402
from models import build_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--run_name", required=True)
    p.add_argument("--arch", default="srcnn_res", choices=["srcnn_res", "deep"])
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--loss", default="mse", choices=["mse", "mse_ssim"])
    p.add_argument("--ssim_weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_ds, val_ds, test_ds = get_datasets(None, None, args.cache_dir, n_train=args.n_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=10, num_workers=args.num_workers)

    model = build_model(**vars(args)).to(device)
    loss_fn = make_loss(args.loss, args.ssim_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Device {device} | model {args.arch} | parameters {count_params(model):,} | loss {args.loss} "
          f"| train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

    start_epoch, best_psnr, history = 1, -1.0, []
    last_path = run_dir / "last.pth"
    if last_path.exists():                                    # resume after a Colab disconnect
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_psnr, history = ckpt["epoch"] + 1, ckpt["best_psnr"], ckpt["history"]
        print(f"Resuming from epoch {start_epoch} (best val PSNR so far {best_psnr:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_psnr, val_ssim, val_mse = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_mse.mean().item(),
               "val_psnr": val_psnr.mean().item(), "val_ssim": val_ssim.mean().item()}
        history.append(row)
        if row["val_psnr"] > best_psnr:
            best_psnr = row["val_psnr"]
            torch.save(model.state_dict(), run_dir / "best.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_psnr": best_psnr, "history": history}, last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch {epoch:3d} | train loss {train_loss:.5f} | val PSNR {row['val_psnr']:.3f} "
              f"SSIM {row['val_ssim']:.4f} | best {best_psnr:.3f} | {time.time() - t0:.1f}s")

    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    test_psnr, test_ssim, test_mse = evaluate(model, test_loader, device)
    test = {"run": args.run_name, "test_psnr": test_psnr.mean().item(), "test_ssim": test_ssim.mean().item(),
            "test_mse": test_mse.mean().item(), "best_val_psnr": best_psnr}
    pd.DataFrame([test]).round(4).to_csv(run_dir / "test.csv", index=False)
    pd.DataFrame({"psnr": test_psnr.numpy(), "mse": test_mse.numpy()}).to_csv(
        run_dir / "test_per_image.csv", index_label="image")
    print("TEST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})


if __name__ == "__main__":
    main()
