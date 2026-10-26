"""
dataset.py - HipMRI 2D slice loading for the VQ-VAE project.

Loads 2D prostate MRI slices stored as Nifti files, scales each slice to
[0, 1], resizes to a fixed shape, and wraps them in PyTorch Datasets and
DataLoaders for the provided train / validate / test splits.

Run directly to audit the data:
    python dataset.py --data_root "path/to/keras_slices_data"

Nifti loading follows Appendix B of the COMP3710 project specification.
"""

import argparse
import os
import re
from collections import defaultdict
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Folder names inside keras_slices_data for each split (image slices only;
# the matching "seg" folders hold segmentation masks, which we don't need).
SPLIT_FOLDERS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test",
}

# Every slice is resized to this (height, width) so batches stack cleanly.
DEFAULT_SHAPE = (256, 128)


def load_slice(path):
    """Load one Nifti slice as a float32 2D NumPy array."""
    image = nib.load(path).get_fdata(caching="unchanged")
    if image.ndim == 3:  # some HipMRI slices carry an extra trailing dim
        image = image[:, :, 0]
    return image.astype(np.float32)


def normalise(image):
    """Min-max scale a slice to [0, 1]; constant slices become all zeros."""
    low, high = image.min(), image.max()
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def patient_id(path):
    """Extract the patient/case ID from a filename, e.g. 'case_004_...' -> '004'.

    Falls back to the filename stem if no case number is found, so the
    overlap check still runs (it will just be per-file).
    """
    name = os.path.basename(path)
    match = re.search(r"case_?(\d+)", name, flags=re.IGNORECASE)
    return match.group(1) if match else name.split(".")[0]


def list_split_files(data_root, split):
    """Return sorted Nifti paths for one split ('train', 'validate', 'test')."""
    folder = os.path.join(data_root, SPLIT_FOLDERS[split])
    files = sorted(glob(os.path.join(folder, "*.nii*")))
    if not files:
        raise FileNotFoundError(f"No Nifti files found in {folder}")
    return files


class HipMRIDataset(Dataset):
    """In-memory dataset of normalised, resized HipMRI 2D slices.

    Each item is a float tensor of shape (1, H, W) with values in [0, 1].
    The 2D slice set is small enough to hold in RAM, which keeps training
    fast and avoids re-reading Nifti files every epoch.
    """

    def __init__(self, files, shape=DEFAULT_SHAPE, augment=False):
        self.files = files
        self.augment = augment
        slices = []
        for path in tqdm(files, desc="Loading slices", leave=False):
            image = torch.from_numpy(normalise(load_slice(path)))
            image = image[None, None]  # (1, 1, H, W) for interpolate
            if tuple(image.shape[-2:]) != tuple(shape):
                image = F.interpolate(image, size=shape, mode="bilinear",
                                      align_corners=False)
            slices.append(image[0])  # back to (1, H, W)
        self.images = torch.stack(slices)  # (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # Light augmentation only: small random intensity jitter.
        # No flips, so generated anatomy keeps its real left/right layout.
        if self.augment:
            image = (image * (1 + 0.05 * (torch.rand(1) - 0.5))).clamp(0, 1)
        return image


def get_dataloaders(data_root, batch_size=32, shape=DEFAULT_SHAPE,
                    num_workers=0, augment_train=False):
    """Build train / validate / test DataLoaders from the provided splits."""
    loaders = {}
    for split in SPLIT_FOLDERS:
        dataset = HipMRIDataset(list_split_files(data_root, split), shape,
                                augment=(augment_train and split == "train"))
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=num_workers,
                                    pin_memory=torch.cuda.is_available())
    return loaders


def check_patient_overlap(data_root):
    """Report any patient IDs that appear in more than one split."""
    split_ids = {split: {patient_id(p) for p in list_split_files(data_root, split)}
                 for split in SPLIT_FOLDERS}
    owners = defaultdict(list)
    for split, ids in split_ids.items():
        for pid in ids:
            owners[pid].append(split)
    leaks = {pid: splits for pid, splits in owners.items() if len(splits) > 1}
    return split_ids, leaks


def audit(data_root):
    """Print the data audit used in the feasibility review."""
    print(f"Data root: {data_root}\n")
    for split in SPLIT_FOLDERS:
        files = list_split_files(data_root, split)
        shapes = defaultdict(int)
        low, high = float("inf"), float("-inf")
        for path in tqdm(files, desc=f"Auditing {split}", leave=False):
            image = load_slice(path)
            shapes[image.shape] += 1
            low, high = min(low, image.min()), max(high, image.max())
        print(f"[{split}] {len(files)} slices")
        print(f"  shapes: {dict(shapes)}")
        print(f"  raw intensity range: {low:.1f} to {high:.1f}")
        print(f"  example file: {os.path.basename(files[0])}")

    split_ids, leaks = check_patient_overlap(data_root)
    print("\nPatients per split:",
          {split: len(ids) for split, ids in split_ids.items()})
    if leaks:
        print(f"WARNING: {len(leaks)} patient IDs appear in multiple splits:")
        for pid, splits in sorted(leaks.items())[:10]:
            print(f"  {pid}: {splits}")
    else:
        print("No patient overlap across splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit HipMRI 2D slices.")
    parser.add_argument("--data_root", required=True,
                        help="Folder containing keras_slices_train/validate/test")
    args = parser.parse_args()
    audit(args.data_root)