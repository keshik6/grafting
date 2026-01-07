# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu


# Import base modules
from PIL import Image
import os
from collections import defaultdict

# Import scientific computing modules
from torch.utils.data import Dataset

"""
ImageNetDataset Pytorch Class
"""
class ImageNetDataset(Dataset):
    def __init__(self, folder_path, num_samples_per_class=None, transform=None,
                txt_mapping_folder_to_class_name='./assets/imagenet/map_clsloc.txt', verbose=True):
        """
        Initialize the dataset.
        
        Args:
            folder_path (str): Path to the folder where images are stored.
            num_samples_per_class (int, optional): Number of samples per class to load. Default is None, which loads all samples.
            transform (callable, optional): Optional transform to be applied on a sample.
            txt_mapping_folder_to_class_name (str, optional): Path to the text file mapping folder names to class names. Default is './assets/imagenet/map_clsloc.txt'.
            verbose (bool, optional): If True, prints out some verbose information. Default is True.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.num_samples_per_class = num_samples_per_class
        self.txt_mapping_folder_to_class_name = txt_mapping_folder_to_class_name
        
        self.class_names = os.listdir(self.folder_path)
        self.class_names.sort()  # Sort
        self._load_imgs()
        self.class_names_to_human_readable_format = self.get_folder_name_to_dict()

        if verbose:
            print(f'num_samples_per_class = {num_samples_per_class}')


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = self.class_list[idx]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        class_label = self.class_names.index(class_name)
        human_readable_class_name = self.class_names_to_human_readable_format[class_name]

        return image, class_label, class_name, human_readable_class_name, img_path
    
    def _load_imgs(self):
        self.class_list = []
        self.image_paths = []

        # Organize image paths by class
        class_images = defaultdict(list)

        for c in self.class_names:
            class_path = os.path.join(self.folder_path, c)

            if self.num_samples_per_class is None:
                file_list = [os.path.join(class_path, i) for i in os.listdir(class_path) if i.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]
            else:
                file_list = [os.path.join(class_path, i) for i in os.listdir(class_path) if i.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))][:self.num_samples_per_class]
            class_images[c].extend(file_list)
        
        # Create round-robin order
        max_len = max(len(files) for files in class_images.values())
        min_len = min(len(files) for files in class_images.values())
        print(max_len, min_len)
        round_robin_paths = []
        round_robin_classes = []

        while True:
            added = False
            for class_name in self.class_names:
                if class_images[class_name]:
                    round_robin_paths.append(class_images[class_name].pop(0))
                    round_robin_classes.append(class_name)
                    added = True
            if not added:
                break

        self.image_paths = round_robin_paths
        self.class_list = round_robin_classes


    def get_folder_name_to_dict(self):
        # Define the file path
        file_path = self.txt_mapping_folder_to_class_name

        # Initialize an empty dictionary
        result_dict = {}

        # Open and read the file
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into columns
                columns = line.strip().split()
                # Extract the key (first column) and the value (last column)
                key = columns[0]
                value = columns[-1]
                # Add to the dictionary
                result_dict[key] = value

        # Issues (crane and maillot are duplicated). See issues here: https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
        return result_dict


if __name__ == '__main__':
    # Simple code to test 
    
    print(f'> Running simple code to test ImageNetDataset class (Training Set)')
    imagenet_object = ImageNetDataset('/data/imagenet/train/', num_samples_per_class=None, verbose=True)
    # print(imagenet_object.__getitem__(0))
    print(f'Total #images = {imagenet_object.__len__()}')

    # ======================
    print(f'> Running simple code to test ImageNetDataset class (Validation Set)')
    imagenet_object = ImageNetDataset('/data/imagenet/val/', num_samples_per_class=None, verbose=True)
    # print(imagenet_object.__getitem__(0))
    print(f'Total #images = {imagenet_object.__len__()}')
