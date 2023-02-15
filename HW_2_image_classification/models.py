import torch.nn as nn
import torch.nn.functional as F


# TODO: add some preprocessing to the graph. For example, resize func


class CNN_MLP_network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # formula [(W−K+2P)/S]+1.
        # W is the input volume - in your case 128
        # K is the Kernel size - in your case 5
        # P is the padding - in your case 0 i believe
        # S is the stride - which you have not provided.

        # MLP part
        self.fc1 = nn.Linear(in_features=34*34*16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=self.num_classes)

    def forward(self, t):
        #Layer 1
        t = t.float()  # 150, 150, 3
        #Layer 2
        t = self.conv1(t)  # 146, 146, 6
        t = F.relu(t)  # 146, 146, 6
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # 73, 73, 6
        #Layer 3
        t = self.conv2(t)  # 69, 69, 16
        t = F.relu(t)  # 69, 69, 16
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # 34, 34, 16
        #Layer 4
        t = t.reshape(-1, 34*34*16)
        t = self.fc1(t)
        t = F.relu(t)
        #Layer 5
        t = self.fc2(t)
        t = F.relu(t)
        #Layer 6/ Output Layer
        t = self.out(t)

        return t


class CNN_network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # MLP part
        self.out = nn.Linear(in_features=34*34*16, out_features=self.num_classes)

    def forward(self, t):
        #Layer 1
        t = t.float()  # 150, 150, 3
        #Layer 2
        t = self.conv1(t)  # 146, 146, 6
        t = F.relu(t)  # 146, 146, 6
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # 73, 73, 6
        #Layer 3
        t = self.conv2(t)  # 69, 69, 16
        t = F.relu(t)  # 69, 69, 16
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # 34, 34, 16
        #Layer 4
        t = t.reshape(-1, 34*34*16)
        t = self.out(t)

        return t
