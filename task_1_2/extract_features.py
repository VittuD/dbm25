import os
import cv2
import torch
import torch.nn as nn
import grayscale_moments
import resnet
import hog

def extract_features(image_path: os.path, model: nn.Module) -> dict:
    """
    Extract features from a single .jpg image.

    Parameters:
        image_path (str): The full path to the image.
        model: The model to use for feature extraction.
        resnet: An object with preprocess_image and extract_layer_features methods.
        hog: An object with a hog method for computing HOG features.
        grayscale_moments: An object with a moments method for computing color moments.

    Returns:
        dict: A dictionary containing the features of the image.
              Returns None if the image is invalid.
    """
    if not image_path.endswith('.jpg'):
        print("Unsupported file format.")
        return None

    # Extract class from the filename
    filename = os.path.basename(image_path)
    parts = filename.split('_')
    if len(parts) < 2:
        print("Filename does not contain a valid class.")
        return None
    class_name = parts[0] + '_' + parts[1]

    image_path = os.path.abspath(image_path)
    print(f'Processing {image_path}')

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load image.")
        return None

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_moments = grayscale_moments.moments(img)
    hog_features = hog.hog(grayscale_img)

    preprocessed_img = resnet.preprocess_image(img)
    resnet_features = resnet.extract_layer_features(
        model, preprocessed_img, layer_names=['avgpool', 'layer3', 'fc']
    )
    avgpool = resnet_features['avgpool']
    layer3 = resnet_features['layer3']
    fc = resnet_features['fc']

    avgpool = avgpool.view(avgpool.size(1))
    avgpool = torch.mean(avgpool.view(-1, 2), dim=1)

    layer3 = torch.mean(layer3.view(-1, 1024, 14 * 14), dim=2)
    layer3 = layer3.view(layer3.size(1))

    fc = fc.view(fc.size(1))

    features_dict = {
        'file_path': image_path,
        'class': class_name,
        'cm': color_moments,
        'hog': hog_features,
        'avgpool': avgpool,
        'layer3': layer3,
        'fc': fc
    }

    return features_dict