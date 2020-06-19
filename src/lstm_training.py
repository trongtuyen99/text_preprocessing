class DataUtils(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).long(), self.y[idx]

dataset = DataUtils(sentence_matrix, Y_train)
trainloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

model = LSTMEbd(len(word2idx), 300, 1, 50, 20)
criterion = torch.nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr)

epochs = 101
from time import time
t1 = time()
for epoch in range(epochs):
    for X, Y in trainloader:
        X = X.long()

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        # CUDA out of memory. ==>
        print(f"loss after {epoch} epochs: {loss}") # todo: test from here
        s = 0
        for X, Y in trainloader:
            X = X.long()
            output = model(X)
            pred = torch.argmax(output, axis=1)
            tmp = torch.sum(pred==Y)
            s += tmp
        print(s)
        s = 0
        for X, Y in testloader:
            X = X.long()
            output = model(X)
            pred = torch.argmax(output, axis=1)
            tmp = torch.sum(pred==Y)
            s += tmp
        print(s)

print(f"training {epochs} epochs done after {time()-t1} s")