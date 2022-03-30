import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, model="vgg"):
        super(EncoderCNN, self).__init__()

        if model == "vgg":
            features = models.vgg16(pretrained=True).features
            self.cnn = nn.Sequential(*list(features))
        elif model == "resnet":
            layers = models.resnet50(pretrained=True).children()
            self.cnn = nn.Sequential(*list(layers)[:-2])

        for param in self.cnn.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        features = self.cnn(images)                                         # (batch, channels, h, w)
        features = features.permute(0, 2, 3, 1)                             # (batch, h, w, channels)
        features = features.view(features.size(0), -1, features.size(-1))   # (batch, h * w, channels)

        return features
