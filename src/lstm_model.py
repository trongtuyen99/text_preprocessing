class LSTMEbd(torch.nn.Module):
    def __init__(self, vocab_size, ebd_dim, num_layers, hidden_size, output_size):
        super(LSTMEbd, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, ebd_dim) # add pad before
        self.lstm = torch.nn.LSTM(ebd_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.4)
        # change
        self.linear = torch.nn.Linear(hidden_size, output_size)
        # self.linear2 = torch.nn.Linear(20, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        output = self.embeddings(x)
        output = self.dropout(output)
        # output, hn = self.rnn(output)
        lstm_out, (ht, ct) = self.lstm(output)
        output = self.linear(ht[-1])
        # output = self.linear2(output)
        output = self.softmax(output)
        return output