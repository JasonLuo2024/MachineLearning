__author__ = "JasonLuo"
import torch.nn as nn
import torchvision.models as models
class DualDensennet169(nn.Module):
    def __init__(self):
        super(DualDensennet169, self).__init__()
        self.model_1 = models.densenet169(pretrained=True)
        self.model_2 = models.densenet169(pretrained=True)
        self.num_features = self.model_1.classifier.in_features

        self.model_1.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )

        self.model_2.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x1, x2):
        output1 = self.model_1(x1)

        output2 = self.model_2(x2)

        output = torch.cat((output1, output2), dim=0)

        return output