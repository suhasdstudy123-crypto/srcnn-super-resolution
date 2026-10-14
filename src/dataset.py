"""DIV2K data pipeline for SRCNN (written by us, not from yjn870).

Pipeline for every image (same as the Stanford project):
    target (HR): center crop 800x800 -> resize to 224x224
    input  (LR): target -> downscale to 112x112 -> upscale back to 224x224
The model learns: blurry 224x224 input -> sharp 224x224 target.

Data used (teacher's instruction: train on 800 images):
    train: DIV2K_train_HR, 800 images (0001-0800)
    val  : 50 of DIV2K_valid_HR (0801-0900), chosen with a fixed seed
    test : the other 50 of DIV2K_valid_HR

Speed: decoding 2K PNGs is slow, so targets are made ONCE and cached as a
uint8 .npy file (N x 224 x 224 x 3). The blurry inputs are made on the fly
from the cached targets (cheap at 224x224), so the interpolation stays a
setting we can change.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

INTERP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


def load_target(path, crop=800, size=224):
    """Open one HR image, center crop to crop x crop, resize to size x size.

    A few DIV2K train images have a side shorter than 800 px. For those we take
    the largest centered square that fits (e.g. 648x648) and resize it to 224 the same way.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    crop = min(crop, w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    return img.resize((size, size), Image.BICUBIC)


def degrade(target, scale=2, interp="bicubic", down="bicubic"):
    """Make the blurry input: shrink by `scale`, then enlarge back to the original size.

    The shrink is always bicubic (`down`), so every method starts from the SAME low-res
    image. Only the enlarging method (`interp`) changes. That keeps baselines fair.
    """
    size = target.size[0]
    small = target.resize((size // scale, size // scale), INTERP[down])
    return small.resize((size, size), INTERP[interp])


def build_cache(img_dir, cache_file, crop=800, size=224):
    """Load every PNG in img_dir, process it, and save all of them in one .npy file.

    If the cache already exists, just load it (this is what makes later runs fast).
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        return np.load(cache_file)
    paths = sorted(Path(img_dir).glob("*.png"))
    assert paths, f"No PNG files found in {img_dir}"
    n_small = sum(min(Image.open(p).size) < crop for p in paths)  # reads only the header, fast
    print(f"{img_dir}: {len(paths)} images, {n_small} smaller than {crop}px (cropped to their short side)")
    arr = np.stack([np.asarray(load_target(p, crop, size)) for p in paths])  # (N, 224, 224, 3) uint8
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    print(f"Cached {len(arr)} images -> {cache_file}")
    return arr


def split_val_test(n, seed=42):
    """Shuffle indices 0..n-1 with a fixed seed; first half = val, second half = test."""
    idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[: n // 2]), np.sort(idx[n // 2:])


def to_tensor(img):
    """PIL image (H, W, 3) uint8 -> float tensor (3, H, W) in [0, 1]."""
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class SRDataset(Dataset):
    """Returns (blurry_input, sharp_target) pairs, both 3 x 224 x 224 in [0, 1]."""

    def __init__(self, targets, scale=2, interp="bicubic"):
        self.targets = targets          # uint8 array (N, 224, 224, 3)
        self.scale = scale
        self.interp = interp

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        target = Image.fromarray(self.targets[i])
        blurry = degrade(target, self.scale, self.interp)
        return to_tensor(blurry), to_tensor(target)


def get_datasets(train_dir, valid_dir, cache_dir, n_train=None, interp="bicubic", seed=42):
    """Build train / val / test datasets.

    n_train: use only the first n_train training images (None = all 800).
             Small values are for quick smoke tests, 60 for the Stanford-size comparison.
    """
    train = build_cache(train_dir, Path(cache_dir) / "train_224.npy")
    valid = build_cache(valid_dir, Path(cache_dir) / "valid_224.npy")
    if n_train is not None:
        train = train[:n_train]
    val_idx, test_idx = split_val_test(len(valid), seed)
    return (
        SRDataset(train, interp=interp),
        SRDataset(valid[val_idx], interp=interp),
        SRDataset(valid[test_idx], interp=interp),
    )


if __name__ == "__main__":
    # Quick self-test on real data:
    #   python src/dataset.py <train_dir> <valid_dir> <cache_dir>
    import sys

    train_ds, val_ds, test_ds = get_datasets(*sys.argv[1:4])
    x, y = train_ds[0]
    print("train / val / test sizes:", len(train_ds), len(val_ds), len(test_ds))
    print("input shape:", tuple(x.shape), "target shape:", tuple(y.shape))
    print("pixel range:", float(x.min()), "to", float(x.max()))
