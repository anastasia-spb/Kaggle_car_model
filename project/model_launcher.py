import torch
from model import AlexNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


class ModelLauncher:
    def __init__(self, device):
        self.model = AlexNet()
        self.device = device
        self.model = self.model.to(device=self.device)  # send the model for training on either cuda or cpu
        # Initialize learning parameters
        self.learning_rate = 1e-4
        self.epochs = 50
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 64

    def train(self, torch_dataset, save_model=False):
        # Split to validation and train sets
        validation_set_size = int(0.2 * len(torch_dataset))
        train_set_size = len(torch_dataset) - validation_set_size

        train_subset, validation_subset = random_split(torch_dataset, [train_set_size, validation_set_size],
                                                       generator=torch.Generator().manual_seed(42))
        # Shuffle data both in train and validation sets
        train_dl = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(validation_subset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            loss_ep = 0
            for batch_idx, (_, data, targets) in enumerate(train_dl):
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                ## Forward Pass
                self.optimizer.zero_grad()
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                loss.backward()
                self.optimizer.step()
                loss_ep += loss.item()
            print(f"Loss in epoch {epoch} :::: {loss_ep / len(train_dl)}")

            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                for batch_idx, (_, data, targets) in enumerate(val_dl):
                    data = data.to(device=self.device)
                    targets = targets.to(device=self.device)
                    ## Forward Pass
                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)
                print(
                    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
                )

            if save_model:
                torch.save(self.model.state_dict(), './model_from_last_train.pth')

    def predict(self, torch_dataset, model_from_file=''):
        if model_from_file:
            self.model.load_state_dict(torch.load(model_from_file))

        submission_dl = DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=False)

        submission_results = []
        img_names_column = []
        for batch_idx, (img_names, data, _) in enumerate(submission_dl):
            ## Forward Pass
            scores = self.model(data.cuda())
            softmax = torch.exp(scores).cpu()
            prob = list(softmax.detach().numpy())
            predictions = np.argmax(prob, axis=1)
            submission_results.append(predictions)
            img_names_column.append(img_names)

        return submission_results, img_names_column

