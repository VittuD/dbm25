import cv2
import torch

# Function to preprocess the image
def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """
    Preprocess the image for ResNet model.
    Args:
        image: A tensor of shape [H, W, C] (height, width, channels)
    Returns:
        A tensor of shape [1, C, 224, 224] (batch size, channels, height, width)
    """
    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert the image to a tensor
    image = torch.tensor(image).permute(2, 0, 1).float()
    # Normalize the image
    image = (image - 127.5) / 127.5
    # Add a batch dimension
    image = image.unsqueeze(0)
    return image

def extract_layer_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_names: list
) -> dict:
    """
    Run x through model, grab the output of specified layers via forward hooks,
    then remove the hooks and return a dictionary mapping layer names to their
    corresponding raw feature tensors.

    Args:
        model:        A PyTorch model (e.g. torchvision.models.resnet50(pretrained=True))
        x:            Input tensor of shape [B, C, H, W] (already preprocessed)
        layer_names:  List of dot-separated layer names as in model.named_modules()

    Returns:
        dict: A mapping from each layer name to the output tensor of that layer.
    """
    features = {}
    handles = []
    
    def get_hook(name):
        def _hook(module, inp, out):
            features[name] = out.detach()
        return _hook

    modules = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in modules:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        handle = modules[layer_name].register_forward_hook(get_hook(layer_name))
        handles.append(handle)

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    return features
