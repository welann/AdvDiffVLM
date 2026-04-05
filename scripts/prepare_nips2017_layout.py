#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the flat NIPS2017 dataset layout for AdvDiffVLM by creating "
            "`clean/sample` and `target/sample` directories compatible with ImageFolder."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/model_weight/NIPS2017"),
        help="Dataset root containing `images.csv` and the flat `images/` directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Flat image directory. Defaults to `<root>/images`.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to `images.csv`. Defaults to `<root>/images.csv`.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Output clean root. Defaults to `<root>/clean/sample`.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Output target root. Defaults to `<root>/target/sample`.",
    )
    parser.add_argument(
        "--cam-dir",
        type=Path,
        default=None,
        help="Optional CAM directory to create. Defaults to `<root>/cam`.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of images to place in `target/sample` when no external target "
            "images are provided. `1.0` means all images."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for target subset sampling.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How files should be materialized into clean/target directories.",
    )
    parser.add_argument(
        "--clean-subset",
        action="store_true",
        help=(
            "If set, `clean/sample` will use the same subset as `target/sample`. "
            "Useful for quick smoke tests because the current `main.py` zips the two loaders."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directories before writing the new layout.",
    )
    return parser.parse_args()


def read_image_ids(csv_path: Path) -> List[str]:
    image_ids: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "ImageId" not in reader.fieldnames:
            raise ValueError(f"`{csv_path}` does not contain an `ImageId` column.")
        for row in reader:
            image_ids.append(row["ImageId"])
    if not image_ids:
        raise ValueError(f"`{csv_path}` does not contain any image rows.")
    return image_ids


def list_flat_images(source_dir: Path) -> List[Path]:
    files = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def validate_source(csv_ids: Sequence[str], source_files: Sequence[Path]) -> None:
    csv_set = set(csv_ids)
    file_set = {path.stem for path in source_files}
    missing = sorted(csv_set - file_set)
    extra = sorted(file_set - csv_set)
    if missing or extra:
        preview_missing = ", ".join(missing[:10])
        preview_extra = ", ".join(extra[:10])
        raise ValueError(
            "Source images and `images.csv` do not match.\n"
            f"missing_in_images={len(missing)} [{preview_missing}]\n"
            f"extra_in_images={len(extra)} [{preview_extra}]"
        )


def select_target_ids(csv_ids: Sequence[str], ratio: float, seed: int) -> List[str]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError("`target_ratio` must be in the interval (0, 1].")
    count = max(1, int(round(len(csv_ids) * ratio)))
    rng = random.Random(seed)
    sampled = list(csv_ids)
    rng.shuffle(sampled)
    selected = sampled[:count]
    selected_set = set(selected)
    return [image_id for image_id in csv_ids if image_id in selected_set]


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with `--overwrite` to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_images(
    image_ids: Iterable[str],
    source_dir: Path,
    output_dir: Path,
    link_mode: str,
) -> None:
    for image_id in image_ids:
        source_path = resolve_source_path(source_dir, image_id)
        destination_path = output_dir / source_path.name
        if link_mode == "symlink":
            os.symlink(source_path.resolve(), destination_path)
        elif link_mode == "hardlink":
            os.link(source_path, destination_path)
        elif link_mode == "copy":
            shutil.copy2(source_path, destination_path)
        else:
            raise ValueError(f"Unsupported link mode: {link_mode}")


def resolve_source_path(source_dir: Path, image_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = source_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find an image file for `{image_id}` under {source_dir}.")


def write_id_manifest(path: Path, image_ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for image_id in image_ids:
            handle.write(f"{image_id}\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_dir = (args.source_dir or (root / "images")).resolve()
    csv_path = (args.csv_path or (root / "images.csv")).resolve()
    clean_dir = (args.clean_dir or (root / "clean" / "sample")).resolve()
    target_dir = (args.target_dir or (root / "target" / "sample")).resolve()
    cam_dir = (args.cam_dir or (root / "cam")).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source image directory does not exist: {source_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"`images.csv` does not exist: {csv_path}")

    csv_ids = read_image_ids(csv_path)
    source_files = list_flat_images(source_dir)
    validate_source(csv_ids, source_files)

    target_ids = select_target_ids(csv_ids, args.target_ratio, args.seed)
    clean_ids = target_ids if args.clean_subset else list(csv_ids)

    ensure_empty_dir(clean_dir, args.overwrite)
    ensure_empty_dir(target_dir, args.overwrite)
    cam_dir.mkdir(parents=True, exist_ok=True)

    materialize_images(clean_ids, source_dir, clean_dir, args.link_mode)
    materialize_images(target_ids, source_dir, target_dir, args.link_mode)

    write_id_manifest(clean_dir.parent / "clean_ids.txt", clean_ids)
    write_id_manifest(target_dir.parent / "target_ids.txt", target_ids)

    print("Prepared NIPS2017 layout successfully.")
    print(f"root          : {root}")
    print(f"source_dir    : {source_dir}")
    print(f"clean_dir     : {clean_dir}")
    print(f"target_dir    : {target_dir}")
    print(f"cam_dir       : {cam_dir}")
    print(f"clean_count   : {len(clean_ids)}")
    print(f"target_count  : {len(target_ids)}")
    print(f"link_mode     : {args.link_mode}")
    print(f"clean_subset  : {args.clean_subset}")
    print()
    print("Notes:")
    print("- `clean/sample` and `target/sample` are now compatible with `torchvision.datasets.ImageFolder`.")
    print("- If you keep the current `main.py`, the effective run length is `min(len(clean), len(target))` because the two dataloaders are zipped.")
    print("- Using a 10% target subset is acceptable for debugging or smoke tests, but it is not a faithful reproduction of the paper's target-image construction.")


if __name__ == "__main__":
    main()
