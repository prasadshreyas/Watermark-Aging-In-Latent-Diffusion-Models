"""
Shreyas Prasad
CS 7180 - Advanced Perception
"""

from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.stats import binomtest

def load_model(model_path, device):
    """
    Load a torch model from a specified path.

    Parameters:
    model_path (str): The file path of the torch model.
    device (str): The device type ('cuda' or 'cpu') on which the model will be loaded.

    Returns:
    torch model: The loaded torch model.
    """
    model = torch.jit.load(model_path).to(device)
    return model

def apply_transformations():
    """
    Apply image transformations for preprocessing before model inference.

    Returns:
    torchvision.transforms.Compose: The composed transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_message(model, image_path, transformations, device):
    """
    Extracts a message from an image using a specified model.

    Parameters:
    model (torch model): The model used for extracting the message.
    image_path (str): The path to the image from which to extract the message.
    transformations (torchvision.transforms.Compose): The transformations to apply to the image.
    device (str): The device type ('cuda' or 'cpu') for processing.

    Returns:
    list: A list of boolean values representing the extracted message.
    """
    img = Image.open(image_path)
    img = transformations(img).unsqueeze(0).to(device)
    msg = model(img)
    return (msg > 0).squeeze().cpu().numpy().tolist()

def msg2str(msg):
    """
    Convert a list of boolean values to a string representation.

    Parameters:
    msg (list of bool): The message as a list of boolean values.

    Returns:
    str: The string representation of the message.
    """
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    """
    Convert a string representation of a message to a list of boolean values.

    Parameters:
    str (str): The string representation of the message.

    Returns:
    list of bool: The message as a list of boolean values.
    """
    return [True if el == '1' else False for el in str]

def compute_difference(message, key):
    """
    Compute the difference between two messages.

    Parameters:
    message (list of bool): The first message as a list of boolean values.
    key (list of bool): The second message (key) as a list of boolean values.

    Returns:
    list of bool: A list representing the difference between the two messages.
    """
    return [message[i] != key[i] for i in range(len(message))]

def calculate_bit_accuracy(diff):
    """
    Calculate the bit accuracy between two messages.

    Parameters:
    diff (list of bool): The difference between the two messages.

    Returns:
    float: The bit accuracy.
    """
    return 1 - sum(diff) / len(diff)

def compute_p_value(diff):
    """
    Compute the p-value for the statistical test of the message difference.

    Parameters:
    diff (list of bool): The difference between the two messages.

    Returns:
    p-value: The p-value from the binomial test.
    """
    return binomtest(len(diff) - sum(diff), len(diff), 0.5, alternative='greater')

if __name__ == "__main__":
    # Parameters
    model_path = "/content/stable_signature/models/dec_48b_whit.torchscript.pt"
    image_path = "/content/image.JPEG"
    device = "cuda"
    key = '111010110101000001010111010011010100010000100111'

    # Process
    model = load_model(model_path, device)
    transformations = apply_transformations()
    bool_msg = extract_message(model, image_path, transformations, device)
    print("Extracted message: ", msg2str(bool_msg))

    bool_key = str2msg(key)
    diff = compute_difference(bool_msg, bool_key)
    bit_acc = calculate_bit_accuracy(diff)
    print("Bit accuracy: ", bit_acc)

    pval = compute_p_value(diff)
    print("p-value of statistical test: ", pval)