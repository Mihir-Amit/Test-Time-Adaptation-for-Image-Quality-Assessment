import torch
from models.network_scunet import SCUNet as net
from utils import utils_image as util
import numpy as np


def denoise_image_direct(input_image, model_name='scunet_color_real_psnr', model_zoo='../Image_Denoising/SCUNet/model_zoo', device=None):
    """
    Denoise an image directly without saving to disk.

    Args:
        model_name (str): Name of the model to use (e.g., 'scunet_color_real_psnr').
        input_image (np.ndarray or torch.Tensor): Input noisy image as a NumPy array or PyTorch tensor.
        model_zoo (str): Directory containing the pre-trained model.
        device (torch.device): Device to run the model on. If None, will default to CUDA if available.

    Returns:
        np.ndarray: Denoised image as a NumPy array.
    """
    # Ensure device is set
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    n_channels = 3
    model_path = f"{model_zoo}/{model_name}.pth"
    model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    model = model.to(device)

    # Prepare the input image
    if isinstance(input_image, np.ndarray):
        img_tensor = util.uint2tensor4(input_image).to(device)
    elif isinstance(input_image, torch.Tensor):
        if input_image.dim() == 3:  # (C, H, W)
            img_tensor = input_image.unsqueeze(0).to(device)  # Add batch dimension
        elif input_image.dim() == 4:  # (B, C, H, W)
            img_tensor = input_image.to(device)
    else:
        raise TypeError("Input image must be a NumPy array or a PyTorch tensor.")

    # Perform denoising
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Convert the denoised image back to a NumPy array
    denoised_image = util.tensor2uint(output_tensor)

    return denoised_image
