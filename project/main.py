import torch
import gc
import os
from torchvision import transforms
from jpeg_dataset import JpgDataset
from custom_transformations import PerImageNormalization
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import AlexCaptchaNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import pandas as pd

CLASS_NUM = 10
IMG_SIZE = 227
IMG_CHANNELS = 3
READ_DATASET = True
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCH_NUM = 50
TRAIN = False


# It might be necessary to increase swap
# before loading whole training dataset: https://askubuntu.com/questions/178712/how-to-increase-swap-space
# with parameters bs=1024 count=136314880.
# It took around 100Gb of Swp
def create_dataset(root_dir, labels_csv='', train=True):
    image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or an ndarray to PIL Image
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    if TRAIN:
        # Load or reload and store dataset
        dataset_name = 'avtovaz_dataset.pt'
        if READ_DATASET:
            torch_dataset = create_dataset('../data/train', '../data/train.csv')
            print("Dataset creation finished.")
            # torch.save(torch_dataset.clone(), dataset_name)
        else:
            torch_dataset = torch.load('avtovaz_dataset.pt')

        # Split to validation and train sets
        validation_set_size = int(0.2 * len(torch_dataset))
        train_set_size = len(torch_dataset) - validation_set_size

        train_subset, validation_subset = random_split(torch_dataset, [train_set_size, validation_set_size], generator=torch.Generator().manual_seed(42))
        # Shuffle data both in train and validation sets
        train_dl = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=True)

        model = AlexCaptchaNet()
        model = model.to(device=device)  # send the model for training on either cuda or cpu

        ## Loss and optimizer
        learning_rate = LEARNING_RATE
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        for epoch in range(EPOCH_NUM):
            loss_ep = 0
            for batch_idx, (_, data, targets) in enumerate(train_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                optimizer.zero_grad()
                scores = model(data)
                loss = criterion(scores, targets)
                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
            print(f"Loss in epoch {epoch} :::: {loss_ep / len(train_dl)}")

            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                for batch_idx, (_, data, targets) in enumerate(val_dl):
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    ## Forward Pass
                    scores = model(data)
                    _, predictions = scores.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)
                print(
                    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
                )

        # Save model
        torch.save(model.state_dict(), './avtovaz_model.pth')
    else:
        model = AlexCaptchaNet()
        model = model.to(device=device)  # send the model for training on either cuda or cpu
        model.load_state_dict(torch.load('avtovaz_model.pth'))

        dataset_for_submission = create_dataset('../data/test', '', False)
        submission_dl = DataLoader(dataset_for_submission, batch_size=BATCH_SIZE, shuffle=False)

        submission_resuls = []
        img_names_column = []
        for batch_idx, (img_names, data, _) in enumerate(submission_dl):
            ## Forward Pass
            scores = model(data.cuda())
            softmax = torch.exp(scores).cpu()
            prob = list(softmax.detach().numpy())
            predictions = np.argmax(prob, axis=1)
            submission_resuls.append(predictions)
            img_names_column.append(img_names)

        sample_submission = pd.read_csv('../data/sample-submission.csv')
        sample_submission['Category'] = np.concatenate(submission_resuls)
        sample_submission['Id'] = np.concatenate(img_names_column)

        sample_submission.to_csv('avtovaz_submission2.csv', index=False)










