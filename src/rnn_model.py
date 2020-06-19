import torch
class RNNMd(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, batch, output_size):
        super(RNNMd, self).__init__()
        self.num_layer = num_layer
        self.batch = batch
        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layer, batch_first=True)
        self.softmax1 = torch.nn.Softmax(dim=1)   
        self.fc = torch.nn.Linear(hidden_size, output_size)
        print("init model done!")
    def forward(self, x):
        """
        x: batch
        """
        # batch = x.size
        # batch_ = x.shape[1] # conflict name rs 
        # print("in forward")
        h0 = torch.zeros(self.num_layer, x.shape[1], self.hidden_size) # init
        # if torch.cuda.is_available():
        #     h0.cuda()
        # print("computing output done!")
        out, hn = self.rnn(x, h0)
        # checkout hn[-1] here instead of out[-1]
        out = hn[-1].contiguous().view(-1, self.hidden_size) # contiguous(): make a copy of out just ff, take last output only
        # out = self.fc(out) 
        out = self.softmax1(out)
        # print("done forward!")
        return out, hn