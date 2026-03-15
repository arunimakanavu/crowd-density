import torch
import torch.nn as nn
from torchvision import models
import openvino as ov
from pathlib import Path


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())

        self.frontend = nn.Sequential(*features[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def convert(checkpoint_path: str, output_dir: str = "assets/models"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSRNet architecture...")
    model = CSRNet()

    print(f"Loading checkpoint from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # dummy input — batch=1, RGB, typical HD frame
    dummy_input = torch.rand(1, 3, 720, 1280)

    print("Converting to OpenVINO IR...")
    ov_model = ov.convert_model(model, example_input=dummy_input)

    output_path = output_dir / "csrnet.xml"
    ov.save_model(ov_model, str(output_path))
    print(f"Saved IR model to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSRNet PyTorch checkpoint to OpenVINO IR")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--output_dir", type=str, default="assets/models", help="Output directory for IR files")
    args = parser.parse_args()

    convert(args.checkpoint, args.output_dir)
