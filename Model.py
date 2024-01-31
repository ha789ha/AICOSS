from torch import nn
import torchvision.models as models

class Efficientnet(nn.Module):
    def __init__(self, num_classes=60):
        super(Efficientnet, self).__init__()
        self.backbone = models.efficientnet_v2_m(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

class Resnet(nn.Module):
    def __init__(self, num_classes=60):
        super(Resnet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
