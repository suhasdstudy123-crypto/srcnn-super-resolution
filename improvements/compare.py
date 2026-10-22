"""Compare the Stanford SRCNN with our improvements: table + figures (written by us).

Reads   <stanford_dir>/main/            history.csv, test.csv, best.pth   (replication run)
        <imp_dir>/imp1_residual/ ...    same files for each improvement run
        <baselines>                     results/baselines.csv (bicubic etc. on the same split)
        <cache_dir>/valid_224.npy       to re-run every model on the 50 test images
        <sweep_csv> (optional)          ssim_weight_sweep.csv from the weight sweep
Writes to <out_dir>:
    improvement_table.csv   test PSNR / SSIM, gain over bicubic and over Stanford, params, time
    ladder.png              test PSNR and test SSIM of every step (bicubic -> Stanford -> imp1-3)
    val_curves.png          val PSNR per epoch for Stanford and the 3 improvements
    per_image_gain.png      gain over bicubic per test image: Stanford vs our best model
    ssim_sweep.png          val PSNR and SSIM for each SSIM weight (if the sweep csv is given)
    demo_<i>.png            Bicubic | Stanford SRCNN | Improved | Original, with a zoom

Run (Colab):
    python improvements/compare.py --stanford_dir <drive>/checkpoints \
        --imp_dir <drive>/improvement-checkpoints --baselines results/baselines.csv \
        --cache_dir <drive>/data/cache --sweep_csv <drive>/improvement-results/ssim_weight_sweep.csv \
        --out_dir <drive>/improvement-results
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import degrade, split_val_test, to_tensor  # noqa: E402
from demo import best_zoom_box, to_img  # noqa: E402
from model import SRCNN, count_params  # noqa: E402
from plots import BLUE, GRAY, INK, MUTED  # noqa: E402  (also applies the replication's plot style)
from utils import psnr_per_image  # noqa: E402

from models import build_model  # noqa: E402

ORANGE, AQUA = "#eb6834", "#1baf7a"
# run folder name -> (label in figures, colour). Stanford is the gray reference.
RUNS = {
    "main": ("Stanford SRCNN", GRAY),
    "imp1_residual": ("imp1 residual", BLUE),
    "imp2_deep10": ("imp2 deep10", ORANGE),
    "imp3_deep10_ssim": ("imp3 deep10 + SSIM", AQUA),
}


def run_dir(args, run):
    return Path(args.stanford_dir if run == "main" else args.imp_dir) / run


def load_model(folder):
    """Rebuild a trained model from its folder (config.json says which model it is)."""
    config = json.loads((folder / "config.json").read_text())
    model = build_model(config["model"]) if "model" in config else SRCNN()   # 'main' has no 'model' key
    model.load_state_dict(torch.load(folder / "best.pth", map_location="cpu"))
    return model.eval()


def make_table(args, runs, bicubic):
    """One row per method: test metrics, gains, parameter count, seconds per epoch."""
    rows = [{"method": "bicubic", "test_psnr": bicubic["psnr"], "test_ssim": bicubic["ssim"]}]
    for run in runs:
        folder = run_dir(args, run)
        test = pd.read_csv(folder / "test.csv").iloc[0]
        history = pd.read_csv(folder / "history.csv")
        rows.append({"method": RUNS[run][0], "test_psnr": test.test_psnr, "test_ssim": test.test_ssim,
                     "best_val_psnr": test.best_val_psnr, "params": count_params(load_model(folder)),
                     "sec_per_epoch": history.seconds.mean() if "seconds" in history else np.nan})
    table = pd.DataFrame(rows)
    table["gain_vs_bicubic_db"] = table.test_psnr - bicubic["psnr"]
    table["gain_vs_stanford_db"] = table.test_psnr - table.set_index("method").test_psnr["Stanford SRCNN"]
    table = table.round(4)
    table["params"] = table["params"].astype("Int64")      # whole numbers; empty for bicubic
    return table


def plot_ladder(table, out_file):
    """Two dot plots side by side (PSNR, SSIM): each step of the ladder, bicubic dashed."""
    steps = table[table.method != "bicubic"].reset_index(drop=True)
    colors = [c for _, c in RUNS.values()][: len(steps)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(steps) + 1.8), sharey=True)
    for ax, col, name, fmt in [(axes[0], "test_psnr", "Test PSNR (dB)", "{:.2f}"),
                               (axes[1], "test_ssim", "Test SSIM", "{:.3f}")]:
        bic = table.set_index("method").loc["bicubic", col]
        ax.scatter(steps[col], steps.method, color=colors, s=80, zorder=3, edgecolor="white", linewidth=2)
        ax.axvline(bic, color=INK, lw=1.5, ls="--")
        ax.text(bic, len(steps) - 0.45, f" bicubic {fmt.format(bic)}", color=INK, fontsize=9)
        for y, v in enumerate(steps[col]):        # label above each dot (the y axis is flipped)
            ax.annotate(fmt.format(v), (v, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=9)
        span = steps[col].max() - min(steps[col].min(), bic)
        ax.set(xlim=(min(steps[col].min(), bic) - 0.15 * span, steps[col].max() + 0.15 * span),
               ylim=(-0.6, len(steps) - 0.2), title=name)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Each improvement added on top of the previous one (50 test images)", color=INK)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_val_curves(args, runs, bicubic_val, out_file):
    """Val PSNR per epoch; each line labelled at its end, so colour is not the only cue."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for run in runs:
        label, color = RUNS[run]
        h = pd.read_csv(run_dir(args, run) / "history.csv")
        ax.plot(h.epoch, h.val_psnr, color=color, lw=2)
        ends.append([h.val_psnr.iloc[-1], f"{label} ({h.val_psnr.max():.2f})", color, h.epoch.iloc[-1]])
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)                            # minimum label spacing: 4.5% of the y range
    for i in range(1, len(ends)):                      # nudge end labels apart so they don't overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, text, color, x in ends:
        ax.text(x + 0.015 * x + 1, y, text, color=INK, fontsize=9, va="center")
        ax.plot([x], [y], "s", color=color, ms=6)
    ax.axhline(bicubic_val, color=INK, lw=1.5, ls="--")
    ax.text(2, bicubic_val + 0.08, f"bicubic {bicubic_val:.2f}", color=INK, fontsize=9)
    ax.set(xlabel="Epoch", ylabel="Val PSNR (dB)", title="Validation PSNR during training (best in brackets)")
    last_epoch = max(x for _, _, _, x in ends)
    ax.set_xlim(0, last_epoch * 1.32)                  # room on the right for the labels
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


@torch.no_grad()
def per_image_psnr(args, runs):
    """Re-run bicubic and every model on the 50 test images. Returns a DataFrame of PSNRs."""
    valid = np.load(Path(args.cache_dir) / "valid_224.npy")
    _, test_idx = split_val_test(len(valid))
    sharp = torch.stack([to_tensor(Image.fromarray(valid[i])) for i in test_idx])
    blurry = torch.stack([to_tensor(degrade(Image.fromarray(valid[i]))) for i in test_idx])
    result = {"bicubic": psnr_per_image(blurry, sharp).numpy()}
    for run in runs:
        model = load_model(run_dir(args, run))
        pred = torch.cat([model(b).clamp(0, 1) for b in blurry.split(10)])
        result[run] = psnr_per_image(pred, sharp).numpy()
    return pd.DataFrame(result), blurry, sharp


def plot_per_image(psnrs, best_run, out_file):
    """Scatter: x = Stanford gain over bicubic, y = our best model's gain. Above the line = we win."""
    x = psnrs["main"] - psnrs.bicubic
    y = psnrs[best_run] - psnrs.bicubic
    wins = int((y > x).sum())
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo, hi = min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--")
    mid = lo + 0.85 * (hi - lo)
    ax.annotate("same gain", (mid, mid), xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
    ax.scatter(x, y, color=RUNS[best_run][1], s=50, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Stanford SRCNN: gain over bicubic (dB)",
           ylabel=f"{RUNS[best_run][0]}: gain over bicubic (dB)",
           title=f"Per test image: {RUNS[best_run][0]} better on {wins} of {len(x)}")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return wins


def plot_sweep(sweep_csv, out_file):
    """Val PSNR and val SSIM for each SSIM weight: two panels, never two y-axes on one chart."""
    s = pd.read_csv(sweep_csv)
    labels = [str(w) for w in s.ssim_weight]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, name, fmt in [(axes[0], "val_psnr", "Val PSNR (dB)", "{:.2f}"),
                               (axes[1], "val_ssim", "Val SSIM", "{:.4f}")]:
        ax.plot(labels, s[col], color=AQUA, lw=2, marker="o", ms=8, mec="white", mew=2)
        for i, v in enumerate(s[col]):
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=9)
        pad = 0.25 * (s[col].max() - s[col].min())
        ax.set(title=name, xlabel="SSIM weight (0 = plain MSE)", ylim=(s[col].min() - pad, s[col].max() + pad))
    fig.suptitle("SSIM-weight sweep: deep10, 10 epochs, mean of epochs 8-10 (val only)", color=INK)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


@torch.no_grad()
def plot_demo(args, improved_run, blurry, sharp, psnrs, index, out_file):
    """Bicubic | Stanford SRCNN | Improved | Original for one test image, with a zoomed box."""
    b, s = blurry[index:index + 1], sharp[index:index + 1]
    stanford = load_model(run_dir(args, "main"))(b).clamp(0, 1)
    improved = load_model(run_dir(args, improved_run))(b).clamp(0, 1)
    panels = [("Bicubic", b, psnrs.bicubic[index]), ("Stanford SRCNN", stanford, psnrs["main"][index]),
              (RUNS[improved_run][0], improved, psnrs[improved_run][index]), ("Original", s, None)]
    top, left = best_zoom_box(stanford, improved, s)     # where ours beats Stanford the most
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for col, (name, x, p) in enumerate(panels):
        img = to_img(x)
        axes[0, col].imshow(img)
        axes[0, col].add_patch(plt.Rectangle((left, top), 56, 56, fill=False, ec="#eda100", lw=2))
        axes[0, col].set_title(name if p is None else f"{name}\n{p:.2f} dB", fontsize=11)
        axes[1, col].imshow(img[top:top + 56, left:left + 56], interpolation="nearest")
        axes[1, col].set_title("zoom", fontsize=9, color=MUTED)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stanford_dir", required=True, help="folder holding the replication run 'main'")
    p.add_argument("--imp_dir", required=True, help="folder holding imp1_residual, imp2_deep10, ...")
    p.add_argument("--baselines", required=True, help="results/baselines.csv")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sweep_csv", default=None)
    p.add_argument("--improved", default="imp3_deep10_ssim", help="run shown in the demo figures")
    p.add_argument("--demo_index", type=int, nargs="+", default=[0, 7, 23], help="test images, 0-49")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = [r for r in RUNS if (run_dir(args, r) / "test.csv").exists()]   # skip runs not finished yet
    print("Runs found:", runs)
    base = pd.read_csv(args.baselines).query("method == 'bicubic'").set_index("split")
    bicubic = {"psnr": base.loc["test", "psnr"], "ssim": base.loc["test", "ssim"]}

    table = make_table(args, runs, bicubic)
    table.to_csv(out / "improvement_table.csv", index=False)
    print(table.to_string(index=False))
    plot_ladder(table, out / "ladder.png")
    plot_val_curves(args, runs, base.loc["val", "psnr"], out / "val_curves.png")

    psnrs, blurry, sharp = per_image_psnr(args, runs)
    psnrs.round(4).to_csv(out / "per_image_psnr.csv", index_label="image")
    for run in runs:                       # re-running the saved weights must reproduce test.csv
        print(f"check {run:18s}: re-computed test PSNR {psnrs[run].mean():.4f}")
    best = max(runs[1:], key=lambda r: psnrs[r].mean())
    wins = plot_per_image(psnrs, best, out / "per_image_gain.png")
    print(f"Best improvement on test PSNR: {best}; beats Stanford on {wins} of {len(psnrs)} test images")

    if args.sweep_csv and Path(args.sweep_csv).exists():
        plot_sweep(args.sweep_csv, out / "ssim_sweep.png")
    improved = args.improved if args.improved in runs else best
    for i in args.demo_index:
        plot_demo(args, improved, blurry, sharp, psnrs, i, out / f"demo_{i}.png")
    print(f"Saved table and figures -> {out}")


if __name__ == "__main__":
    main()
