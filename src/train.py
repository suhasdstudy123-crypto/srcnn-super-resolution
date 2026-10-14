"""Train SRCNN on DIV2K (written by us; loop structure inspired by yjn870/SRCNN-pytorch train.py).

Each epoch: train on all training images, then score on the 50 val images.
Saves to <out_dir>/<run_name>/:
    history.csv   one row per epoch: train loss, val loss, val PSNR, val SSIM
    best.pth      weights with the highest val PSNR (used for test + demo)
    last.pth      latest weights + optimizer state, so a run can resume after Colab disconnects
    test.csv      final test metrics of best.pth (mean PSNR / SSIM / MSE)
    test_per_image.csv  PSNR and MSE of every test image (for the loss-vs-PSNR question)

Run (Stanford settings):
    python src/train.py --cache_dir <cache> --out_dir <drive>/checkpoints --run_name main
Ablations: add --lr 0.01, --weight_decay 1e-4, --tanh_out, or --use_bn.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import SRCNN
from utils import psnr_per_image, set_seed, ssim_per_image


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """One pass over the training set. Returns the mean training loss (MSE)."""
    model.train()
    total, count = 0.0, 0
    for blurry, sharp in loader:
        blurry, sharp = blurry.to(device), sharp.to(device)
        loss = loss_fn(model(blurry), sharp)   # how far the output is from the sharp target
        optimizer.zero_grad()                  # clear old gradients
        loss.backward()                        # compute new gradients
        optimizer.step()                       # update the weights
        total += loss.item() * len(blurry)
        count += len(blurry)
    return total / count


@torch.no_grad()  # no gradients needed when only scoring
def evaluate(model, loader, device):
    """Score the model on a dataset. Returns per-image PSNR, SSIM and MSE tensors."""
    model.eval()
    psnrs, ssims, mses = [], [], []
    for blurry, sharp in loader:
        blurry, sharp = blurry.to(device), sharp.to(device)
        pred = model(blurry).clamp(0, 1)
        psnrs.append(psnr_per_image(pred, sharp).cpu())
        ssims.append(ssim_per_image(pred, sharp))
        mses.append(((pred - sharp) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(psnrs), torch.cat(ssims), torch.cat(mses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--train_dir", default=None, help="only needed if the cache does not exist yet")
    p.add_argument("--valid_dir", default=None, help="only needed if the cache does not exist yet")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--run_name", default="main")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--use_bn", action="store_true")
    p.add_argument("--tanh_out", action="store_true")
    p.add_argument("--n_train", type=int, default=None, help="use only the first N train images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Device: {device} | run dir: {run_dir}")

    # ----- data -----
    train_ds, val_ds, test_ds = get_datasets(args.train_dir, args.valid_dir, args.cache_dir, n_train=args.n_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=10, num_workers=args.num_workers)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # ----- model, loss, optimizer -----
    model = SRCNN(use_bn=args.use_bn, tanh_out=args.tanh_out).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ----- resume if a previous run of the same name was cut off -----
    start_epoch, best_psnr, history = 1, -1.0, []
    last_path = run_dir / "last.pth"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_psnr, history = ckpt["epoch"] + 1, ckpt["best_psnr"], ckpt["history"]
        print(f"Resuming from epoch {start_epoch} (best val PSNR so far {best_psnr:.3f})")

    # ----- training loop -----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_psnr, val_ssim, val_mse = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_mse.mean().item(),
            "val_psnr": val_psnr.mean().item(),
            "val_ssim": val_ssim.mean().item(),
        }
        history.append(row)

        if row["val_psnr"] > best_psnr:                 # keep the best epoch, judged on VAL only
            best_psnr = row["val_psnr"]
            torch.save(model.state_dict(), run_dir / "best.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_psnr": best_psnr, "history": history}, last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(f"epoch {epoch:3d} | train loss {train_loss:.5f} | val PSNR {row['val_psnr']:.3f} "
              f"SSIM {row['val_ssim']:.4f} | best {best_psnr:.3f} | {time.time() - t0:.1f}s")

    # ----- final test, ONCE, with the best-on-val weights -----
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    test_psnr, test_ssim, test_mse = evaluate(model, test_loader, device)
    test = {"run": args.run_name, "test_psnr": test_psnr.mean().item(),
            "test_ssim": test_ssim.mean().item(), "test_mse": test_mse.mean().item(),
            "best_val_psnr": best_psnr}
    pd.DataFrame([test]).round(4).to_csv(run_dir / "test.csv", index=False)
    pd.DataFrame({"psnr": test_psnr.numpy(), "mse": test_mse.numpy()}).to_csv(
        run_dir / "test_per_image.csv", index_label="image")
    print("TEST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})


if __name__ == "__main__":
    main()
