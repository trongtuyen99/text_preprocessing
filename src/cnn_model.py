import torch
class CNNModel(torch.nn.Module):
    def __init__(self, input_dim, n_class):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=2) # 24 input
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1) # 32 input
        self.relu2 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # 16 input
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1) # 24 input
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=4, stride=4) # 8 input

        self.fc1 = torch.nn.Linear(input_dim * 6, out_features=n_class)
        self.activ1 = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.relu1(out)
        out = self.conv2(out)
        # print(out.shape)
        out = self.relu2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        # print(out.shape)
        out = self.maxpool2(out)
        # print(out.shape)
        out = self.fc1(out.view(out.shape[0], -1))
        out = self.activ1(out)
        
        return out