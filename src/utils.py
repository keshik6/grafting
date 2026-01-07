# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

import yaml
import hashlib
import torch
import matplotlib.pyplot as plt
import logging
from PIL import Image
import pandas as pd

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

# Add utils for training/ evals.
def generate_sha_hash(file_paths):
    # Sort the list to ensure deterministic hash
    sorted_file_paths = sorted(file_paths)
    # Convert the sorted list to a single string
    file_paths_str = ''.join(sorted_file_paths)
    # Generate the SHA256 hash
    sha_hash = hashlib.sha256(file_paths_str.encode()).hexdigest()
    return sha_hash


def save_hash(hash_key, file_name='hash_key.txt'):
    with open(file_name, 'w') as file:
        file.write(hash_key)

def load_hash(file_name='hash_key.txt'):
    with open(file_name, 'r') as file:
        return file.read().strip()

def verify_hash(current_hash, saved_hash):
    return current_hash == saved_hash


def remove_keys(original_dict, keys_to_remove):
    return {k: v for k, v in original_dict.items() if k not in keys_to_remove}

def log_worker_shards(url):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        print(f"Worker {worker_info.id} processing shard: {url}")
    return url


def plot_histogram(data_dict, output_filename='histogram_with_table.png'):
    """
    Plot histogram from dictionary. We use this function as a sanuty check for Stratified splits.
    """
    # Convert bytes keys to strings
    #data_dict = {k.decode('utf-8'): v for k, v in data_dict.items()}
    
    # Convert the dictionary to a DataFrame for better handling
    df = pd.DataFrame(list(data_dict.items()), columns=['Name', 'Count'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram
    ax.hist(df['Count'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Counts')
    
    # Add a table to the plot
    table_data = df['Count'].value_counts().reset_index().sort_values(by='index')
    table_data.columns = ['Value', 'Count']
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Find the keys with the top 2 highest counts
    top_keys = df.nlargest(3, 'Count')
    # Print the keys with the top 2 highest counts
    print("Keys with the top 2 highest counts:")
    for _, row in top_keys.iterrows():
        print(f"{row['Name']}: {row['Count']}")
    
    # Adjust layout to make room for the table
    plt.subplots_adjust(right=0.75)
    
    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()



#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger