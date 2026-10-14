"""Make the report figures and the results table from the saved run folders (written by us).

Reads   <ckpt_dir>/<run>/history.csv and test.csv for every run, plus baselines.csv
Writes  <out_dir>/results_table.csv   one row per method/run: test PSNR, SSIM, gain over bicubic
        <out_dir>/training_curve.png  main run: train loss and val PSNR per epoch
        <out_dir>/ablations.png       test PSNR of every run vs the bicubic baseline

Run: python src/plots.py --ckpt_dir <drive>/checkpoints --baselines <drive>/results/baselines.csv --out_dir results
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import pandas as pd

BLUE, GRAY, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "figure.dpi": 150,
})


def load_runs(ckpt_dir):
    """Collect test.csv of every finished run (skips the 'smoke' test run)."""
    rows = []
    for test_file in sorted(Path(ckpt_dir).glob("*/test.csv")):
        if test_file.parent.name != "smoke":
            rows.append(pd.read_csv(test_file).iloc[0])
    return pd.DataFrame(rows)


def results_table(runs, baselines):
    """Baselines + all runs in one table, with the gain over bicubic on the same test set."""
    base = baselines[baselines.split == "test"].rename(
        columns={"method": "run", "psnr": "test_psnr", "ssim": "test_ssim", "mse": "test_mse"})
    table = pd.concat([base[["run", "test_psnr", "test_ssim", "test_mse"]],
                       runs[["run", "test_psnr", "test_ssim", "test_mse", "best_val_psnr"]]])
    bicubic = base.set_index("run").loc["bicubic", "test_psnr"]
    table["gain_vs_bicubic_db"] = table["test_psnr"] - bicubic
    return table.round(4), bicubic


def plot_training_curve(history, bicubic_val, out_file):
    """Two panels sharing the epoch axis: loss (left) and val PSNR (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(history.epoch, history.train_loss, color=BLUE, lw=2)
    ax1.set(title="Training loss (MSE)", xlabel="Epoch")
    ax2.plot(history.epoch, history.val_psnr, color=BLUE, lw=2, label="SRCNN")
    ax2.axhline(bicubic_val, color=GRAY, lw=2, ls="--", label=f"Bicubic ({bicubic_val:.2f} dB)")
    best = history.loc[history.val_psnr.idxmax()]
    ax2.plot(best.epoch, best.val_psnr, "o", color=BLUE, ms=8, mec="white", mew=2,
             label=f"Best {best.val_psnr:.2f} dB (epoch {int(best.epoch)})")
    ax2.set(title="Validation PSNR (dB)", xlabel="Epoch")
    ax2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_ablations(runs, bicubic, out_file):
    """Dot plot of test PSNR (dots, not bars, because the axis does not start at 0).

    The main run is blue, the rest gray; bicubic is the dashed reference line.
    """
    runs = runs.sort_values("test_psnr").reset_index(drop=True)
    colors = [BLUE if r == "main" else GRAY for r in runs.run]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(runs) + 1.4))
    ax.scatter(runs.test_psnr, runs.run, color=colors, s=70, zorder=3)
    ax.axvline(bicubic, color=INK, lw=1.5, ls="--")
    ax.text(bicubic, -0.7, f"bicubic {bicubic:.2f} ", color=INK, fontsize=9, ha="right", va="center")
    for y, v in enumerate(runs.test_psnr):
        ax.text(v + 0.08, y, f"{v:.2f}", va="center", color=INK, fontsize=9)
    lo = min(runs.test_psnr.min(), bicubic) - 0.8
    ax.set(xlim=(lo, runs.test_psnr.max() + 0.6), ylim=(-1, len(runs) - 0.5),
           xlabel="Test PSNR (dB)", title="Test PSNR by experiment")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baselines = pd.read_csv(args.baselines)
    runs = load_runs(args.ckpt_dir)
    table, bicubic = results_table(runs, baselines)
    table.to_csv(out / "results_table.csv", index=False)
    print(table.to_string(index=False))

    bicubic_val = baselines.query("split == 'val' and method == 'bicubic'").psnr.item()
    history = pd.read_csv(Path(args.ckpt_dir) / "main" / "history.csv")
    plot_training_curve(history, bicubic_val, out / "training_curve.png")
    plot_ablations(runs, bicubic, out / "ablations.png")
    print(f"\nSaved table and figures -> {out}")


if __name__ == "__main__":
    main()
