import cv2
import torch
import numpy as np

def blockify(
    image: torch.Tensor, 
    new_H: int, 
    new_W: int, 
    block_H: int, 
    block_W: int
) -> torch.Tensor:
    """
    Break a grayscale image into non-overlapping blocks of size block_H×block_W.

    Args:
        image:   torch.Tensor of shape (H, W) or NumPy array (H, W)
        new_H:   desired output height (must be divisible by block_H)
        new_W:   desired output width  (must be divisible by block_W)
        block_H: block height
        block_W: block width

    Returns:
        torch.Tensor of shape (N, block_H, block_W), where
        N = (new_H // block_H) * (new_W // block_W)
    """
    # convert to NumPy for cv2 if needed
    np_img = image.numpy() if isinstance(image, torch.Tensor) else image

    # resize: cv2.resize expects (width, height)
    resized = cv2.resize(np_img, (new_W, new_H))
    H, W = resized.shape

    # ensure exact divisibility
    assert H % block_H == 0 and W % block_W == 0, (
        f"Resized ({H}×{W}) must be multiples of block size ({block_H}×{block_W})"
    )

    blocks = []
    # slide a window of size block_H×block_W
    for top in range(0, H, block_H):
        for left in range(0, W, block_W):
            patch = resized[top: top + block_H,
                            left: left + block_W]   # shape (block_H, block_W)
            # back to torch
            blocks.append(torch.from_numpy(patch).float())

    # stack: (N, block_H, block_W)
    return torch.stack(blocks, dim=0)

def moment(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the first 3 moments of the image, mean, variance, and skewness.

    Args:
        image: Input image

    Returns:
        Tuple: Moments of the image
    """
    # Convert to NumPy for calculations
    image = image.numpy() if isinstance(image, torch.Tensor) else image
    # Calculate the mean
    mean = np.mean(image)
    # Calculate the variance
    variance = np.var(image)
    # Calculate the skewness
    skewness = np.mean((image - mean) ** 3) / (variance ** 1.5) if variance != 0 else 0
    return torch.tensor([mean, variance, skewness])

def moments(image: torch.Tensor) -> torch.Tensor:
    # Blockify the image into 30*10 blocks
    blocks = blockify(image, 300, 100, 30, 10)
    
    # Initialize list to store moments
    moments = []
    
    # Calculate the moments for each block
    for block in blocks:
        if block is None:
            print("Block is None")
        # Calculate the moments for each block
        block_moments = moment(block)
        # Append the moments to the list
        moments.append(block_moments)
    # Stack the moments into a tensor
    moments = torch.stack(moments)
    # Now from (100*3) to (300,)
    moments = moments.view(-1)
    return moments