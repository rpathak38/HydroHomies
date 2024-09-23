from torch.utils.data import Dataset
import os
from PIL import Image


class FloodDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        super(FloodDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms

        # Paths to the flooding and normal folders
        self.flood_path = os.path.join(self.data_path, "flooding")
        self.normal_path = os.path.join(self.data_path, "normal")

        # Store full paths to images and corresponding labels
        self.flood_data = [os.path.join(self.flood_path, f) for f in os.listdir(self.flood_path)]
        self.normal_data = [os.path.join(self.normal_path, n) for n in os.listdir(self.normal_path)]

        # Combine flood and normal data and create labels
        self.data = self.flood_data + self.normal_data
        self.labels = [1] * len(self.flood_data) + [0] * len(self.normal_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and corresponding label
        img_path = self.data[idx]
        label = self.labels[idx]
        print("YOUOUOUOU")
        print(img_path)
        # Open the image as a PIL image
        image = Image.open(img_path).convert('RGB')

        # Apply the transformations (e.g., resize, normalize, etc.)
        if self.transforms:
            image = self.transforms(image)

        return image, label
