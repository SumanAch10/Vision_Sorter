"""Run the foundation on synthetic images before adding a real dataset."""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_sorter.model import TinySorterCNN
from vision_sorter.preprocess import load_image


def main() -> None:
    image_path = ROOT / "data" / "foundation_sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (240, 180), color=(40, 160, 80)).save(image_path)

    image = load_image(image_path).unsqueeze(0)
    model = TinySorterCNN()
    logits = model(image)
    predicted_class = logits.argmax(dim=1).item()

    print(f"image tensor shape: {tuple(image.shape)}")
    print(f"model output shape: {tuple(logits.shape)}")
    print(f"predicted class id before training: {predicted_class}")
    print("foundation smoke test passed")


if __name__ == "__main__":
    main()
