from torch.nn import Linear, Conv2d, MaxPool2d, Module, BatchNorm2d
import torch.nn.functional as functional


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4,
                            padding=0)
        self.batchnorm1 = BatchNorm2d(96)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv2d(in_channels=96,
                            out_channels=256,
                            kernel_size=5, stride=1,
                            padding=2)
        self.batchnorm2 = BatchNorm2d(256)

        self.conv3 = Conv2d(in_channels=256,
                            out_channels=384,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.batchnorm3 = BatchNorm2d(384)

        self.conv4 = Conv2d(in_channels=384,
                            out_channels=384,
                            kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = BatchNorm2d(384)

        self.conv5 = Conv2d(in_channels=384,
                            out_channels=256,
                            kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = BatchNorm2d(256)

        self.fc1 = Linear(in_features=9216,
                          out_features=4096)
        self.fc2 = Linear(in_features=4096,
                          out_features=4096)
        self.fc3 = Linear(in_features=4096, out_features=10)

    # Forward pass
    def forward(self, x):
        x = functional.relu(self.conv1(x.float()))
        x = self.batchnorm1(x)
        x = self.maxpool(x)

        x = functional.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.maxpool(x)

        x = functional.relu(self.conv3(x))
        x = self.batchnorm3(x)

        x = functional.relu(self.conv4(x))
        x = self.batchnorm4(x)

        x = functional.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
