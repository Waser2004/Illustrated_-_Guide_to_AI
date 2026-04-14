import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys

SHARED_MODEL_DIR = Path(__file__).resolve().parents[1] / "Cube Detection model"
if str(SHARED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_MODEL_DIR))

from model import build_simple_cnn_regressor
from utils import load_image

# Device (use CUDA when available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load the trained model checkpoint
    checkpoint_path = SHARED_MODEL_DIR / "checkpoint.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_simple_cnn_regressor(bounded_output=True, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Get the convolutions of the first layer
    first_conv_layer = model.encoder.features[0]
    conv_weights = first_conv_layer.weight.data.cpu().numpy()

    # -----------------------------------------------------------------
    # Visualize the filters of the first convolutional layer
    # ------------------------------------------------------------------
    num_filters = conv_weights.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(num_filters):
        filter_weights = conv_weights[i]
        filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min())
        axes[i].imshow(filter_weights.transpose(1, 2, 0))
        axes[i].axis("off")
        axes[i].set_title(f"fil. {i + 1}")

    plt.savefig(Path(__file__).resolve().parent / "filters.png")
    plt.show()

    # -----------------------------------------------------------------
    # Visualize the activations of the first convolutional layer
    # -----------------------------------------------------------------
    image_path = SHARED_MODEL_DIR / "example.png"
    image = load_image(image_path, checkpoint=checkpoint).to(device)

    with torch.no_grad():
        activations = first_conv_layer(image).cpu().numpy()

    num_activations = activations.shape[1]
    fig, axes = plt.subplots(1, num_activations, figsize=(15, 5))
    for i in range(num_activations):
        activation_map = activations[0, i]
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        axes[i].imshow(activation_map, cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"act. {i + 1}")

    plt.savefig(Path(__file__).resolve().parent / "activations.png")
    plt.show()

if __name__ == "__main__":
    main()
