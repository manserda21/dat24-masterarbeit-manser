# train_keypoint_heatmap_unet.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================

class HeatmapDataset(Dataset):

    def __init__(
        self,
        image_dir,
        heatmap_dir,
    ):
        self.image_dir = Path(image_dir)
        self.heatmap_dir = Path(heatmap_dir)

        self.image_files = sorted(
            list(self.image_dir.glob("*.PNG"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = self.image_files[idx]

        stem = image_path.stem

        heatmap_path = self.heatmap_dir / f"{stem}.npz"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (320, 180),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        heatmaps = np.load(heatmap_path)["heatmaps"]

        image = torch.tensor(
            image,
            dtype=torch.float32,
        )

        heatmaps = torch.tensor(
            heatmaps,
            dtype=torch.float32,
        )

        return image, heatmaps


# ============================================================
# U-Net Blocks
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)




class UNet(nn.Module):

    def __init__(
        self,
        num_keypoints,
    ):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            256,
            128,
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            128,
            64,
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            64,
            32,
        )

        self.final = nn.Conv2d(
            32,
            num_keypoints,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)

        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# Training
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    running_loss = 0.0

    for images, heatmaps in tqdm(loader):

        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            heatmaps,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    for images, heatmaps in loader:

        images = images.to(device)
        heatmaps = heatmaps.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            heatmaps,
        )

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        dataset_dir / "keypoint_metadata.json"
    ) as f:
        metadata = json.load(f)

    train_dataset = HeatmapDataset(
        dataset_dir / "images/train",
        dataset_dir / "heatmaps/train",
    )

    val_dataset = HeatmapDataset(
        dataset_dir / "images/val",
        dataset_dir / "heatmaps/val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    num_keypoints = metadata["num_keypoints"]
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Num keypoints: {num_keypoints}")
    print(f"Device: {device}")

    model = UNet(
        num_keypoints=num_keypoints,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    for epoch in range(args.epochs):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"| train={train_loss:.6f} "
            f"| val={val_loss:.6f}"
        )

        torch.save(
            model.state_dict(),
            output_dir / "last_model.pt",
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )

            print(
                f"New best model "
                f"({best_loss:.6f})"
            )

    print()
    print("Training finished")
    print(f"Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()