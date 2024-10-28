import os
from torch.utils.data import Dataset
from PIL import Image
import random
import re

class TemporalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.dataset_size = len(self.image_paths) - 1

    def _get_image_paths(self):
        # retrieve all image paths and sort by time
        image_paths = sorted([
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
            if f.endswith('.tif')
        ])
        return image_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        start_image_path = self.image_paths[idx]
        end_image_path = self.image_paths[idx + 1]

        # extract timestamps from filenames
        start_time = self._extract_time(start_image_path)
        end_time = self._extract_time(end_image_path)
        time_skip = abs(end_time - start_time)

        start_image = Image.open(start_image_path)
        end_image = Image.open(end_image_path)

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)
    
        return start_image, end_image, time_skip

    def _extract_time(self, image_path):
        match = re.search(r'_([0-9]{4}_[0-9]{2})_', image_path)
        if match:
            year, month = map(int, match.group(1).split('_'))
            return year * 12 + month  # converting to months
        return 0
