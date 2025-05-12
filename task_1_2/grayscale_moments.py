import cv2
import torch
import numpy as np

def blockify(image: torch.Tensor, new_H: int, new_W: int, block_H: int, block_W: int) -> torch.Tensor:
    """
    Blockify the image into non-overlapping blocks of size block_H x block_W.
    
    Args:
        image: A tensor of shape [H, W, C] (height, width, channels)
    Returns:
        A tensor of shape [N, block_H, block_W, C] where N is the number of blocks
    """
    image = cv2.resize(image, (new_H, new_W))
    blocks = []
    # Divide the image into non-overlapping 30*10 blocks
    for i in range(0, image.shape[0], block_W):
        for j in range(0, image.shape[1], block_H):
            block = image[i:i+block_W, j:j+block_H].T  # Transpose the block to get shape (30, 10)
            # To torch tensor
            block = torch.tensor(block, dtype=torch.float32)
            blocks.append(block)

    return torch.stack(blocks) if blocks else torch.empty(0)

def moments(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the first 3 moments of the image, mean, variance, and skewness.

    Args:
        image: Input image

    Returns:
        Tuple: Moments of the image
    """
    # Calculate the mean
    mean = np.mean(image)
    # Calculate the variance
    variance = np.var(image)
    # Calculate the skewness
    skewness = np.mean((image - mean) ** 3) / (variance ** 1.5) if variance != 0 else 0
    return torch.tensor([mean, variance, skewness])
