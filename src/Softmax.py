import torch
from torch.autograd import Variable
from torchsummary import summary
from time import time

class SoftmaxMs(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super(SoftmaxMs, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, 200)
    self.activ1 = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(200, 100)
    self.activ2 = torch.nn.ReLU()
    self.linear3 = torch.nn.Linear(100, output_dim)
    self.activ3 = torch.nn.Sigmoid()

    self.activend = torch.nn.Softmax(dim=1)

    self.batchnorm2 = torch.nn.BatchNorm1d(200)
    self.dropout1 = torch.nn.Dropout(0.4)
    self.dropout2 = torch.nn.Dropout(0.3)

  def forward(self, x):
    output = self.linear1(x)
    output = self.activ1(output)
    output = self.dropout1(output)

    output = self.linear2(output)
    output = self.activ2(output)
    output = self.dropout2(output)

    output = self.linear3(output)
    output = self.activ3(output)
    output = self.activend(output)

    return output
