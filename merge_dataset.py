"""
merge_dataset.py

Splits the labeled frames in phase2_dataset/ into train/valid and copies
them into your existing dataset's folder structure (train/images,
train/labels, valid/images, valid/labels). Doesn't touch test/ or your
existing train/valid frames -- only adds to them.

Validation frames are deliberately drawn from both the "easy" clips and the
harder opposite-angle ones (serve4/serve5 by default), so your validation
score actually reflects performance on the case you're trying to fix,
instead of only the easier angle.

Usage:
    python merge_dataset.py --phase2_dir ./phase2_dataset \
        --dataset_root ./path/to/your/downloaded/dataset --valid_count 8
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase2_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path,
                         help="Folder containing train/, valid/, test/, data.yaml")
    parser.add_argument("--valid_count", type=int, default=8)
    parser.add_argument("--priority_prefixes", default="serve4,serve5",
                         help="Comma-separated filename prefixes to make sure "
                              "are represented in the validation split "
                              "(default: the opposite-angle clips)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = args.phase2_dir / "images"
    labels_dir = args.phase2_dir / "labels"
    names = sorted(p.stem for p in images_dir.glob("*.jpg"))
    if not names:
        print(f"No images found in {images_dir}")
        return

    prefixes = tuple(p.strip() for p in args.priority_prefixes.split(",") if p.strip())
    priority_names = [n for n in names if n.startswith(prefixes)]
    other_names = [n for n in names if not n.startswith(prefixes)]
    random.shuffle(priority_names)
    random.shuffle(other_names)

    half = args.valid_count // 2
    remainder = args.valid_count - half
    valid_names = set(priority_names[:remainder] + other_names[:half])

    if len(valid_names) < args.valid_count:
        leftover = [n for n in (priority_names + other_names) if n not in valid_names]
        random.shuffle(leftover)
        for n in leftover:
            if len(valid_names) >= args.valid_count:
                break
            valid_names.add(n)

    train_names = [n for n in names if n not in valid_names]

    train_img_dir = args.dataset_root / "train" / "images"
    train_lbl_dir = args.dataset_root / "train" / "labels"
    valid_img_dir = args.dataset_root / "valid" / "images"
    valid_lbl_dir = args.dataset_root / "valid" / "labels"
    for d in (train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(name, img_dst, lbl_dst):
        shutil.copy2(images_dir / f"{name}.jpg", img_dst / f"{name}.jpg")
        label_src = labels_dir / f"{name}.txt"
        if label_src.exists():
            shutil.copy2(label_src, lbl_dst / f"{name}.txt")
        else:
            (lbl_dst / f"{name}.txt").write_text("")

    for n in train_names:
        copy_pair(n, train_img_dir, train_lbl_dir)
    for n in sorted(valid_names):
        copy_pair(n, valid_img_dir, valid_lbl_dir)

    print(f"Copied {len(train_names)} frames into train/, "
          f"{len(valid_names)} into valid/")
    print("Validation frames:")
    for n in sorted(valid_names):
        print(f"  {n}")


if __name__ == "__main__":
    main()
