import torch.nn as nn

from gaze_estimation.modules import resnet50,resnet34,resnet18

class gaze_network(nn.Module):
    def __init__(self):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet18(pretrained=True)
        self.gaze_fc = nn.Sequential(
            nn.Linear(512, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze
