import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# U-Net
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, num_keypoints):
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

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2,
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2,
        )

        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(
            32,
            num_keypoints,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)

        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# Helpers
# ============================================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    original = image.copy()

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    h_orig, w_orig = image.shape[:2]

    image = cv2.resize(
        image,
        (320, 180),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = torch.tensor(
        image,
        dtype=torch.float32,
    ).unsqueeze(0)

    return image, original, w_orig, h_orig


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.metadata) as f:
        metadata = json.load(f)

    num_keypoints = metadata["num_keypoints"]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet(
        num_keypoints=num_keypoints
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.model,
            map_location=device,
        )
    )

    model.eval()

    image_tensor, image_vis, w_orig, h_orig = load_image(
        args.image
    )

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        heatmaps = torch.sigmoid(
            output
        )[0].cpu().numpy()

    scale_x = w_orig / 320.0
    scale_y = h_orig / 180.0

    predictions = []

    for kp_id, kp_info in metadata["keypoints"].items():

        idx = kp_info["index"]

        heatmap = heatmaps[idx]

        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape,
        )

        confidence = float(
            heatmap[y, x]
        )

        x_img = float(x * scale_x)
        y_img = float(y * scale_y)

        predictions.append({
            "keypoint_id": kp_id,
            "confidence": confidence,
            "x": x_img,
            "y": y_img,
            "world_x": kp_info["world_x"],
            "world_y": kp_info["world_y"],
        })

        cv2.circle(
            image_vis,
            (int(x_img), int(y_img)),
            6,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            image_vis,
            kp_id,
            (int(x_img) + 5, int(y_img) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    with open(
        output_dir / "predicted_keypoints.json",
        "w",
    ) as f:
        json.dump(
            predictions,
            f,
            indent=2,
        )

    cv2.imwrite(
        str(output_dir / "predicted_keypoints.png"),
        image_vis,
    )

    print()
    print("Prediction finished")
    print(f"Keypoints: {len(predictions)}")
    print(f"Output: {output_dir}")
    print()


if __name__ == "__main__":
    main()