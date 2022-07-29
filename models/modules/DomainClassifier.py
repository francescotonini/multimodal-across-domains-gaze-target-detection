import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.alpha is not None:
            output = grad_output.neg() * ctx.alpha
        else:
            output = grad_output.neg()

        return output, None


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.relu = nn.ReLU(True)
        self.softmax = nn.LogSoftmax(dim=1)

        self.fc1 = nn.Linear(1024, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, head_features, alpha):
        head_features = ReverseLayerF.apply(head_features, alpha)

        head_feat_reduced = self.avgpool(head_features).view(-1, 1024)

        x = self.fc1(head_feat_reduced)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
