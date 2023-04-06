import torch
import torch.nn as nn


# define the neural network
class QNetwork(nn.Module):
    def __init__(self, state_space_shape, action_space_shape):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=state_space_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=action_space_shape)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x
    

# define the RND parameters
def feat_extractor(state_space_shape):
    feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels=state_space_shape, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7*7*64, 512),
        nn.ReLU(),
    )
    return feature_extractor


class RNDModel(nn.Module):
    def __init__(self, feature_extractor):
        super(RNDModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.predictor(x)


