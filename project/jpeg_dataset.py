import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os


class JpgDataset(Dataset):
    def __init__(self, images_dir: str, labels_csv='', transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_data = self.__read_images(images_dir)
        if labels_csv:
            self.labels = pd.read_csv(labels_csv)
        else:
            self.labels = None

    def __len__(self):
        return len(self.img_data)

    def __getlabel(self, img_name: str):
        if self.labels is not None:
            sample = self.labels[self.labels['Id'] == img_name]
            if not sample.empty:
                return sample.Category.values[0].item()
        return 0

    def __read_images(self, dir_name: str):
        img_data = []
        for img_name in os.listdir(dir_name):
            with Image.open(os.path.join(dir_name, img_name)).convert('RGB') as sample_image:

                img_data.append((img_name, np.asarray(sample_image)))
        return img_data

    def __getitem__(self, index):
        img_name, image = self.img_data[index]
        if self.transform is not None:
            image = self.transform(image)
        return img_name, image, self.__getlabel(img_name)
