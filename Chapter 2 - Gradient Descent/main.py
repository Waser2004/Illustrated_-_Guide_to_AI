import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

from utils import load_image
from model import build_simple_cnn_regressor

# Device (use CUDA when available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_direction(params, seed = None) -> OrderedDict[str, torch.Tensor]:
    """
    Generate random direction vectors for each parameter tensor.
    
    Args:
        params: List of (name, parameter) tuples from model.named_parameters() 
        seed: Random seed for reproducibility
    Returns:
        direction: OrderedDict mapping parameter names to random direction tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    direction = OrderedDict()
    for name, param in params:
        if param.requires_grad:
            direction[name] = torch.randn_like(param.data)

    return direction


def normalize_direction(direction, params) -> OrderedDict[str, torch.Tensor]:
    """
    Normalize the direction tensors to match the norm of each parameter tensor.

    Args:
        direction: OrderedDict mapping parameter names to direction tensors
        params: List of (name, parameter) tuples from model.named_parameters()

    Returns:
        normalized_direction: OrderedDict with normalized direction tensors
    """
    param_dict = OrderedDict(params)
    normalized_direction = OrderedDict()

    for name, dir_tensor in direction.items():
        param_norm = torch.norm(param_dict[name].data)
        dir_norm = torch.norm(dir_tensor)

        # Avoid division by zero
        if dir_norm > 0:
            normalized_direction[name] = dir_tensor * (param_norm / dir_norm)
        else:
            normalized_direction[name] = dir_tensor

    return normalized_direction

def main(num_points=25, alpha_range=(-2.5, 2.5), beta_range=(-2.5, 2.5)):
    # Load the trained model checkpoint
    checkpoint_path = Path(__file__).resolve().parent / "checkpoint.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_simple_cnn_regressor(bounded_output=True, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load the example image
    LABEL = 0.566605
    image_path = Path(__file__).resolve().parent / "example.png"
    image = load_image(image_path, checkpoint)
    image = image.to(device)

    # get the model's parameters (after moving model to device)
    original_params = OrderedDict()
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()

    filtered_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    # Generate two random directions and normalize them
    seed_1 = 420
    seed_2 = 69

    direction_1 = get_random_direction(filtered_params, seed=seed_1)
    direction_2 = get_random_direction(filtered_params, seed=seed_2)
    direction_1 = normalize_direction(direction_1, filtered_params)
    direction_2 = normalize_direction(direction_2, filtered_params)

    # Prepare label tensor and sample points in the alpha-beta plane
    label_tensor = torch.tensor(LABEL, device=device)

    alphas       = np.linspace(alpha_range[0], alpha_range[1], num_points)
    betas        = np.linspace(beta_range[0], beta_range[1], num_points)
    loss_surface = []

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            loss_surface.append([])

            for j, beta in enumerate(betas):
                # Create a copy of the model's parameters
                for (name, param), dir1, dir2 in zip(filtered_params, direction_1.values(), direction_2.values()):
                    param.data = original_params[name] + alpha * dir1 + beta * dir2

                # Compute the loss for the current alpha-beta pair
                output = model(image)
                loss = torch.nn.functional.mse_loss(output.squeeze(), torch.tensor(LABEL).to(device))
                loss_surface[-1].append(loss.item())
            
        # restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]

    # Create loss surface plot
    losses = np.array(loss_surface)

    fig, ax = plt.subplots(figsize=(10, 8))
    contourf = ax.contourf(alphas, betas, losses, 10, cmap='viridis', alpha=0.8) 
    # contour  = ax.contour(alphas, betas, losses, 10, colors='white', linewidths=0.5)

    # Save loss surface plot as PNG in the same directory
    save_path = Path(__file__).resolve().parent / "loss_surface.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main(num_points=512)