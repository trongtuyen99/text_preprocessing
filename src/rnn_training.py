import torch 
from torch.utils.data import Dataset, DataLoader
class DataUtils(Dataset):
    def __init__(self, X, Y):
        # x, y, z = X.shape # not reshape like bro thought
        # X_reshape = np.zeros((y, x, z))
        # for i in range(X.shape[0]):
        #     X_reshape[:, i, :] = X[i]

        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(Y)
        self.length = self.x.shape[0]

    def __getitem__(self, index):    
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length

dataset = DataUtils(matrix_train, Y_train)
trainloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

torch.cuda.empty_cache()

input_size, hidden_size, num_layer, batch, output_size = 50, 40, 2, 128, 20
model = RNNMd(input_size, hidden_size, num_layer, batch, output_size)

if torch.cuda.is_available():
    pass
    # model.cuda()

print("done init model cuda")
criterion = torch.nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr)

epochs = 11
from time import time
t1 = time()
for epoch in range(epochs):
    for X, Y in trainloader:
        # x, y, z = X.shape # not reshape like bro thought
        # X_reshape = np.zeros((y, x, z))

        # for i in range(X.shape[0]):
        #     X_reshape[:, i, :] = X[i]

        X_reshape = torch.from_numpy(X_reshape).float()
        # if torch.cuda.is_available():
        #     # X = Variable(X.view(X.shape[1], X.shape[0], X.shape[2]).cuda())
        #     X = Variable(X_reshape)
        #     # Y = Variable(Y.cuda())
        #     Y = Variable(Y)
        # else:
        X = Variable(X)
        Y = Variable(Y)
        
        optimizer.zero_grad()
        output, hn = model(X)
        # print(f"shape of X: {X.shape}")
        # print(Y.shape)
        # print(X_reshape.shape)
        # print(output.shape)
        # break
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        # CUDA out of memory. ==>
        print(f"loss after {epoch} epochs: {loss}") # todo: test from here
        # pred1 = torch.argmax(model(torch.from_numpy(X_train[:1000]).view(X_train[:1000].shape[1], X_train[0], X_train[:1000].shape[2]).float().cuda()), axis=1)
        # s1 = sum(pred1 == torch.from_numpy(Y_train[:1000]).cuda())
        # print(f"true train: {s1}")

        # pred2 = torch.argmax(model(torch.from_numpy(X_test).view(X_test.shape[0], 1, X_test.shape[1]).float().cuda()), axis=1)
        # s2 = sum(pred2 == torch.from_numpy(Y_test_transform).cuda())
        # print(f"true test: {s2}")

print(f"training {epochs} epochs done after {time()-t1} s")