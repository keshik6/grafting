import yaml
import hashlib
import torch
import matplotlib.pyplot as plt
import logging
from PIL import Image

# ------------- HELPER FUNCTIONS -------------
def load_yaml_file(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def display_image(path, title=None, figsize=(8, 8)):
    """
    Display a saved image from disk.
    
    Args:
        path (str): Path to the image file (e.g., 'sample.png').
        title (str): Optional title for the plot.
        figsize (tuple): Size of the display figure.
    """
    img = Image.open(path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')                      # Remove axis ticks and labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove borders
    plt.margins(0)                       # Remove margins
    plt.show()
