from torch.utils.data import Dataset, DataLoader
class DataUtils(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_train).float()
        self.y = torch.from_numpy(Y_train_transform)
        self.length = self.x.shape[0]

    def __getitem__(self, index):    
        return self.x[index],self.y[index]

    def __len__(self):
        return self.length

dataset = DataUtils()
trainloader=DataLoader(dataset=dataset,batch_size=128, shuffle=True)

torch.cuda.empty_cache()
model = CNNModel(1000, 20)
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr)

epochs = 101
from time import time
t1 = time()
for epoch in range(epochs):
    for X, Y in trainloader:
        if torch.cuda.is_available():
            X = Variable(X.view(X.shape[0], 1, X.shape[1]).cuda())
            Y = Variable(Y.cuda())
        else:
            X = Variable(X.shape[0], 1, X.shape[1])
            Y = Variable(Y)
        
        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        # CUDA out of memory. ==>
        print(f"loss after {epoch} epochs: {loss}")
        pred1 = torch.argmax(model(torch.from_numpy(X_train[:1000]).view(X_train[:1000].shape[0], 1, X_train[:1000].shape[1]).float().cuda()), axis=1)
        s1 = sum(pred1 == torch.from_numpy(Y_train_transform[:1000]).cuda())
        print(f"true train: {s1}")

        # pred2 = torch.argmax(model(torch.from_numpy(X_test).view(X_test.shape[0], 1, X_test.shape[1]).float().cuda()), axis=1)
        # s2 = sum(pred2 == torch.from_numpy(Y_test_transform).cuda())
        # print(f"true test: {s2}")

print(f"training {epochs} epochs done after {time()-t1} s")