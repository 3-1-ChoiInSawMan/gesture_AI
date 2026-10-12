import argparse
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = "dataset"
DEFAULT_TARGET = "dataset_noisy"
HAND_FEATURES = 84


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a noisy augmented copy of a gesture .npy dataset."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source dataset root. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target dataset root. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Number of noisy samples to create per source file. Default: 1",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation. Default: 0.01",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation. Default: 42",
    )
    parser.add_argument(
        "--max-total-per-class",
        type=int,
        default=None,
        help=(
            "Optional cap for total files per class in the target directory. "
            "Example: 240"
        ),
    )
    parser.add_argument(
        "--include-shoulders",
        action="store_true",
        help="Also add noise to the last 4 shoulder features. Default: hands only",
    )
    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Only write noisy files, without copying original .npy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the target directory when names already exist.",
    )
    return parser.parse_args()


def load_sequence(path: Path) -> np.ndarray:
    seq = np.load(path)
    if seq.ndim != 2:
        raise ValueError(f"{path} must have shape (T, D), got {seq.shape}")
    if seq.shape[1] != 88:
        raise ValueError(f"{path} must have 88 features, got {seq.shape[1]}")
    return seq.astype(np.float32)


def add_gaussian_noise(
    seq: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    include_shoulders: bool,
) -> np.ndarray:
    noisy = seq.copy()
    end = noisy.shape[1] if include_shoulders else HAND_FEATURES
    noise = rng.normal(loc=0.0, scale=noise_std, size=noisy[:, :end].shape)
    noisy[:, :end] += noise.astype(np.float32)
    return noisy.astype(np.float32)


def save_array(path: Path, array: np.ndarray, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {path}")
    np.save(path, array.astype(np.float32))


def copy_original(src: Path, dst: Path, overwrite: bool):
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst}")
        dst.unlink()
    shutil.copy2(src, dst)


def main():
    args = parse_args()

    if args.copies < 1:
        raise ValueError("--copies must be at least 1")
    if args.noise_std <= 0:
        raise ValueError("--noise-std must be greater than 0")
    if args.max_total_per_class is not None and args.max_total_per_class < 1:
        raise ValueError("--max-total-per-class must be at least 1")

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if source == target:
        raise ValueError("Source and target must be different directories.")

    rng = np.random.default_rng(args.seed)
    target.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {source}")

    total_originals = 0
    total_noisy = 0

    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            continue

        out_dir = target / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        originals_written = 0
        noisy_written = 0
        base_count = 0 if args.no_copy_originals else len(files)
        max_noisy = len(files) * args.copies

        if args.max_total_per_class is not None:
            max_noisy = min(max_noisy, max(0, args.max_total_per_class - base_count))

        for src_file in files:
            if not args.no_copy_originals:
                copy_original(src_file, out_dir / src_file.name, args.overwrite)
                originals_written += 1

            seq = load_sequence(src_file)
            for copy_idx in range(1, args.copies + 1):
                if noisy_written >= max_noisy:
                    break
                noisy = add_gaussian_noise(
                    seq=seq,
                    rng=rng,
                    noise_std=args.noise_std,
                    include_shoulders=args.include_shoulders,
                )
                out_name = f"{src_file.stem}_noise{copy_idx:02d}.npy"
                save_array(out_dir / out_name, noisy, args.overwrite)
                noisy_written += 1

            if noisy_written >= max_noisy:
                continue

        total_originals += originals_written
        total_noisy += noisy_written
        class_total = originals_written + noisy_written
        print(
            f"{class_dir.name}: originals={originals_written}, "
            f"noisy={noisy_written}, total={class_total}"
        )

    print(
        f"Done. copied_originals={total_originals}, "
        f"created_noisy={total_noisy}, target={target}"
    )


if __name__ == "__main__":
    main()
