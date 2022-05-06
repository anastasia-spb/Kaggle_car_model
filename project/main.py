import torch
import gc
import os
from torchvision import transforms
from jpeg_dataset import JpgDataset
from custom_transformations import PerImageNormalization
import numpy as np
import pandas as pd
from model_launcher import ModelLauncher


# Dataset parameters
CLASS_NUM = 10
IMG_SIZE = 227
IMG_CHANNELS = 3
# If True, then read raw dataset, transform and form Torch Dataset.
# Otherwise, read torch dataset
READ_DATASET = True
# Train model if True. Load model from file otherwise.
TRAIN = False


# It might be necessary to increase swap
# before loading whole training dataset: https://askubuntu.com/questions/178712/how-to-increase-swap-space
# with parameters bs=1024 count=136314880.
# It took around 100Gb of Swp
def create_dataset(root_dir, labels_csv='', train=True):
    image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or a ndarray to PIL Image
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        PerImageNormalization()])

    if train:
        datasets = []
        for dir_name in os.listdir(root_dir):
            datasets.append(JpgDataset(os.path.join(root_dir, dir_name), labels_csv, image_transformation))
            print('Images from dir with label', dir_name, 'loaded.')

        return torch.utils.data.ConcatDataset(datasets)
    else:
        return JpgDataset(root_dir, labels_csv, image_transformation)


def get_dataset():
    # Load or reload and store dataset
    dataset_name = 'avtovaz_dataset.pt'
    if READ_DATASET:
        dataset = create_dataset('../data/train', '../data/train.csv')
        print("Dataset creation finished.")
        # torch.save(torch_dataset.clone(), dataset_name)
    else:
        dataset = torch.load('avtovaz_dataset.pt')

    return dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    model_launcher = ModelLauncher(device)
    dataset_for_submission = create_dataset('../data/test', '', False)

    if TRAIN:
        train_dataset = get_dataset()
        model_launcher.train(train_dataset)
        submission_results, img_names_column = model_launcher.predict(dataset_for_submission)
    else:
        submission_results, img_names_column = model_launcher.predict(dataset_for_submission, './avtovaz_model.pth')

    # Write results into submission file
    sample_submission = pd.read_csv('../data/sample-submission.csv')
    sample_submission['Category'] = np.concatenate(submission_results)
    sample_submission['Id'] = np.concatenate(img_names_column)

    sample_submission.to_csv('avtovaz_submission2.csv', index=False)










